#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>

#define CHECK_CUDA(x) do{ cudaError_t s=(x); if(s!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(s)); \
  std::exit(1);} }while(0)
#define CHECK_CUSPARSE(x) do{ cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuSPARSE error %s:%d: %d\n",__FILE__,__LINE__,(int)s); \
  std::exit(1);} }while(0)

static std::vector<int32_t> read_i32_list(const std::string& path){
    std::ifstream f(path);
    if(!f){ std::perror(("open "+path).c_str()); std::exit(1); }
    std::vector<int32_t> v; v.reserve(1024);
    long long x;
    while(f >> x) v.push_back((int32_t)x);
    return v;
}
static std::vector<float> read_f32_list(const std::string& path){
    std::ifstream f(path);
    if(!f){ std::perror(("open "+path).c_str()); std::exit(1); }
    std::vector<float> v; v.reserve(1024);
    double x;
    while(f >> x) v.push_back((float)x);
    return v;
}
static void write_i32_list(const std::string& path, const std::vector<int32_t>& v){
    std::ofstream f(path);
    for(size_t i=0;i<v.size();++i){ f<<v[i]<<'\n'; }
}
static void write_f32_list(const std::string& path, const std::vector<float>& v){
    std::ofstream f(path);
    for(size_t i=0;i<v.size();++i){
        f.setf(std::ios::fmtflags(0), std::ios::floatfield);
        f.precision(9);
        f<<v[i]<<'\n';
    }
}

struct CSR {
    int32_t rows{}, cols{-1}, nnz{};
    std::vector<int32_t> indptr, indices;
    std::vector<float>   data;
};

static CSR read_csr_txt_prefix(const std::string& prefix){
    CSR M;
    M.indptr  = read_i32_list(prefix + "_indptr.txt");
    M.indices = read_i32_list(prefix + "_indices.txt");
    M.data    = read_f32_list(prefix + "_data.txt");
    if(M.indptr.empty()){
        std::fprintf(stderr,"empty indptr: %s\n", prefix.c_str());
        std::exit(1);
    }
    if(M.indices.size()!=M.data.size()){
        std::fprintf(stderr,"indices/data length mismatch for prefix %s\n", prefix.c_str());
        std::exit(1);
    }
    M.rows = (int32_t)M.indptr.size()-1;
    M.nnz  = (int32_t)M.indices.size();

    M.cols = -1;
    return M;
}

static void write_csr_txt_prefix(const std::string& prefix, const CSR& M){
    write_i32_list(prefix + "_indptr.txt",  M.indptr);
    write_i32_list(prefix + "_indices.txt", M.indices);
    write_f32_list(prefix + "_data.txt",    M.data);
}

static void validate_csr_indices(const CSR& M, const char* name){
    if(M.cols < 0){
        std::fprintf(stderr,"[%s] cols not set before validation\n", name);
        std::exit(1);
    }
    if(!M.indices.empty()){
        int32_t mx = *std::max_element(M.indices.begin(), M.indices.end());
        if(mx >= M.cols){
            std::fprintf(stderr,"[%s] index out of bounds: max index %d >= ncols %d\n",
                         name, mx, M.cols);
            std::exit(1);
        }
        if(*std::min_element(M.indices.begin(), M.indices.end()) < 0){
            std::fprintf(stderr,"[%s] negative column index detected\n", name);
            std::exit(1);
        }
    }
    if((int)M.indptr.size() != M.rows + 1){
        std::fprintf(stderr,"[%s] indptr length %zu != rows+1 (%d)\n",
                     name, M.indptr.size(), M.rows+1);
        std::exit(1);
    }
}

int main(int argc, char** argv){
    if(argc<4){
        std::fprintf(stderr,"Usage: %s A_prefix B_prefix C_prefix\n", argv[0]);
        return 2;
    }
    std::string Apre=argv[1], Bpre=argv[2], Cpre=argv[3];

    CSR A = read_csr_txt_prefix(Apre);
    CSR B = read_csr_txt_prefix(Bpre);


    // A: rows = len(indptr)-1, cols = B.rows
    // B: rows = len(indptr)-1, cols = A.rows
    if(A.rows <= 0 || B.rows <= 0){
        std::fprintf(stderr,"invalid rows: A.rows=%d B.rows=%d\n", A.rows, B.rows);
        return 1;
    }
    A.cols = B.rows;
    B.cols = A.rows;


    validate_csr_indices(A, "A");
    validate_csr_indices(B, "B");

    // device buffers
    int32_t *dAptr,*dAind,*dBptr,*dBind,*dCptr,*dCind;
    float   *dAval,*dBval,*dCval;
    CHECK_CUDA(cudaMalloc(&dAptr,(A.rows+1)*sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dAind,A.nnz*sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dAval,A.nnz*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dBptr,(B.rows+1)*sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dBind,B.nnz*sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dBval,B.nnz*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dAptr,A.indptr.data(),(A.rows+1)*sizeof(int32_t),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dAind,A.indices.data(),A.nnz*sizeof(int32_t),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dAval,A.data.data(),   A.nnz*sizeof(float),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBptr,B.indptr.data(),(B.rows+1)*sizeof(int32_t),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBind,B.indices.data(),B.nnz*sizeof(int32_t),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBval,B.data.data(),   B.nnz*sizeof(float),   cudaMemcpyHostToDevice));

    cusparseHandle_t h; CHECK_CUSPARSE(cusparseCreate(&h));
    CHECK_CUSPARSE(cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_HOST));

    cusparseSpMatDescr_t dA,dB,dC;
    CHECK_CUSPARSE(cusparseCreateCsr(&dA, A.rows, A.cols, A.nnz,
        dAptr,dAind,dAval, CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&dB, B.rows, B.cols, B.nnz,
        dBptr,dBind,dBval, CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUDA(cudaMalloc(&dCptr, (A.rows+1)*sizeof(int32_t)));
    CHECK_CUSPARSE(cusparseCreateCsr(&dC, A.rows, B.cols, 0,
        dCptr, nullptr, nullptr,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    float alpha=1.0f, beta=0.0f;
    cusparseOperation_t opA=CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB=CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType=CUDA_R_32F;
    cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_ALG1;

    cusparseSpGEMMDescr_t desc; CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&desc));

    size_t buf1=0, buf2=0; void *dBuf1=nullptr, *dBuf2=nullptr;
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(h,opA,opB,&alpha,dA,dB,&beta,dC,
        computeType,alg,desc,&buf1,nullptr));
    if(buf1>0) CHECK_CUDA(cudaMalloc(&dBuf1, buf1));
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(h,opA,opB,&alpha,dA,dB,&beta,dC,
        computeType,alg,desc,&buf1,dBuf1));

    CHECK_CUSPARSE(cusparseSpGEMM_compute(h,opA,opB,&alpha,dA,dB,&beta,dC,
        computeType,alg,desc,&buf2,nullptr));
    if(buf2>0) CHECK_CUDA(cudaMalloc(&dBuf2, buf2));
    CHECK_CUSPARSE(cusparseSpGEMM_compute(h,opA,opB,&alpha,dA,dB,&beta,dC,
        computeType,alg,desc,&buf2,dBuf2));

    int64_t Cr,Cc,Cnnz; CHECK_CUSPARSE(cusparseSpMatGetSize(dC,&Cr,&Cc,&Cnnz));
    CHECK_CUDA(cudaMalloc(&dCind, Cnnz*sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dCval, Cnnz*sizeof(float)));
    CHECK_CUSPARSE(cusparseCsrSetPointers(dC, dCptr, dCind, dCval));
    CHECK_CUSPARSE(cusparseSpGEMM_copy(h,opA,opB,&alpha,dA,dB,&beta,dC,
        computeType,alg,desc));

    CSR C; C.rows=(int32_t)Cr; C.cols=(int32_t)Cc; C.nnz=(int32_t)Cnnz;
    C.indptr.resize(C.rows+1); C.indices.resize(C.nnz); C.data.resize(C.nnz);
    CHECK_CUDA(cudaMemcpy(C.indptr.data(), dCptr, (C.rows+1)*sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C.indices.data(),dCind, C.nnz*sizeof(int32_t),      cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C.data.data(),   dCval, C.nnz*sizeof(float),        cudaMemcpyDeviceToHost));

    write_csr_txt_prefix(Cpre, C);
    std::cout<<"[C++] ALG1 wrote "<<Cpre<<"_* (rows="<<C.rows<<", cols="<<C.cols<<", nnz="<<C.nnz<<")\n";

    if(dBuf1) cudaFree(dBuf1);
    if(dBuf2) cudaFree(dBuf2);
    cudaFree(dAptr); cudaFree(dAind); cudaFree(dAval);
    cudaFree(dBptr); cudaFree(dBind); cudaFree(dBval);
    cudaFree(dCptr); cudaFree(dCind); cudaFree(dCval);
    cusparseSpGEMM_destroyDescr(desc);
    cusparseDestroySpMat(dA); cusparseDestroySpMat(dB); cusparseDestroySpMat(dC);
    cusparseDestroy(h);
    return 0;
}
