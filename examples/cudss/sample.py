import cupy as cp
from cupy_backends.cuda.libs.cudss import Handle, Config, Data
from cupy_backends.cuda.libs import cudss 

def build_example1_system():
    n = 5           # 例: Example 1 が 5x5 なら
    nnz = 8
    nrhs = 1
    

    # rowOffsets, colIndices, values は **ホスト側** でまず定義
    # 例: int rowOffsets_host[] = {0, ...};  をそのまま移植
    row_offsets_host = [0, 2, 4, 6, 7, 8]
    col_indices_host = [0, 2, 1, 2, 2, 4, 3, 4]
    values_host = [4.0, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0]
    b_host = [7.0, 12.0, 25.0, 4.0, 13.0]


    # CuPy のデバイス配列にコピー
    row_offsets = cp.array(row_offsets_host, dtype=cp.int32)
    col_indices = cp.array(col_indices_host, dtype=cp.int32)
    values = cp.array(values_host, dtype=cp.float64)
    b = cp.array(b_host, dtype=cp.float64)
    x = cp.zeros_like(b)

    return n, nnz, row_offsets, col_indices, values, b, x


def solve_with_cudss():
    # 1. 例の線形方程式 Ax = b を準備
    n, nnz, row_offsets, col_indices, values, b, x = build_example1_system()

    # 行列の形状など
    nrows = n
    ncols = n
    nrhs = 1       # 右辺・解ベクトルの本数（Example 1 は 1 本想定）

    # cuDSS のハンドル・オブジェクト（仮の API 名）
    print("step0")

    handle = Handle()
    config = Config()
    data = Data()
    A_mat = None
    b_mat = None
    x_mat = None

    try:
        # ---------------------------------------------------------------------
        # cuDSS オブジェクト作成
        # ---------------------------------------------------------------------
        # TODO: 実際の cudss.pyx での関数名に合わせて変更
        print("step1")
        cudss.create(handle)

        print("step1")
        cudss.configCreate(config)
        print("step1")
        data = cudss.dataCreate(handle)

        print("step1")
        # CSR 行列 A のラッパ
        #
        # C 版:
        #   cudssMatrixCreateCsr(&A, nrows, ncols, nnz,
        #                        rowOffsets, NULL, colIndices, values,
        #                        CUDA_R_32I, CUDA_R_64F, mtype, mview, base);
        #
        # という感じなので、Python 版もそれに近いシグネチャを仮定。
        #
        # ポインタは CuPy 配列の .data.ptr（int）で渡すパターンを想定。
        #
        # なお: 実際のシグネチャは cudss.pxd / cudss.pyx に従って修正してください。
        A_mat = cudss.cudssMatrixCreateCsr(
            nrows, ncols, nnz,
            row_offsets.data.ptr,
            None,  # row indices for COO が不要なら None / 0 など
            col_indices.data.ptr,
            values.data.ptr,
            cudss.CUDA_R_32I,   # TODO: あなたの enum/定数名に合わせる
            cudss.CUDA_R_64F,
            cudss.CUDSS_MATRIX_TYPE_GENERAL,   # or SYMMETRIC/PD など
            cudss.CUDSS_MATRIX_VIEW_FULL,
            cudss.CUDSS_INDEX_BASE_ZERO,       # rowOffsets が 0 始まりなら
        )

        # Dense 行列 (ベクトルも "n x nrhs の dense 行列" として扱う)
        print("step2")
        b_mat = cudss.cudssMatrixCreateDn(
            nrows, nrhs,
            b.data.ptr,
            cudss.CUDA_R_64F,
        )
        x_mat = cudss.cudssMatrixCreateDn(
            nrows, nrhs,
            x.data.ptr,
            cudss.CUDA_R_64F,
        )

        # ---------------------------------------------------------------------
        # オプション設定（例：reordering algorithm）
        # ---------------------------------------------------------------------
        print("step3")
        reorder_alg = cudss.CUDSS_ALG_DEFAULT  # TODO: enum の実体に合わせる
        cudss.cudssConfigSet(
            config,
            cudss.CUDSS_REORDERING_ALG,
            reorder_alg,
            cp.int32().itemsize,  # or sizeof(cudssAlgType_t)
        )

        # 必要であればストリームも設定
        # stream = cp.cuda.Stream()
        # cudss.cudssSetStream(handle, stream.ptr)

        print("step4")
        # ---------------------------------------------------------------------
        # フェーズ 1: 解析（reordering & symbolic factorization）
        # ---------------------------------------------------------------------
        cudss.cudssExecute(
            handle,
            cudss.CUDSS_PHASE_ANALYSIS,
            config,
            data,
            A_mat,
            x_mat,
            b_mat,
        )

        # ---------------------------------------------------------------------
        # フェーズ 2: 数値因数分解
        # ---------------------------------------------------------------------
        cudss.cudssExecute(
            handle,
            cudss.CUDSS_PHASE_FACTORIZATION,
            config,
            data,
            A_mat,
            x_mat,
            b_mat,
        )

        # ---------------------------------------------------------------------
        # フェーズ 3: 解く
        # ---------------------------------------------------------------------
        cudss.cudssExecute(
            handle,
            cudss.CUDSS_PHASE_SOLVE,
            config,
            data,
            A_mat,
            x_mat,
            b_mat,
        )

        # ここまで来たら、解は x (CuPy 配列) に入っているはず
        cp.cuda.runtime.deviceSynchronize()
        print("solution x =", x.get())

    finally:
        # ---------------------------------------------------------------------
        # 後始末（実際の API 名に合わせて直してください）
        # ---------------------------------------------------------------------
        if A_mat is not None:
            cudss.cudssMatrixDestroy(A_mat)
        if x_mat is not None:
            cudss.cudssMatrixDestroy(x_mat)
        if b_mat is not None:
            cudss.cudssMatrixDestroy(b_mat)

        if data is not None and handle is not None:
            cudss.dataDestroy(handle, data)

        if config is not None:
            cudss.configDestroy(config)

        if handle is not None:
            cudss.destroy(handle)


if __name__ == "__main__":
    solve_with_cudss()
