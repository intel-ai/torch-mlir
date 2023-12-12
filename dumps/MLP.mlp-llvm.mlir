module attributes {torch.debug_module_name = "MLP"} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func private @refbackend_consume_func_return_mrf32(%arg0: i64, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr)> 
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %2, %4 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @_mlir_ciface_refbackend_consume_func_return_mrf32(%4) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_refbackend_consume_func_return_mrf32(!llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @xsmm_unary_invoke(i64, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64) attributes {sym_visibility = "private"}
  llvm.func @xsmm_binary_invoke(i64, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64) attributes {sym_visibility = "private"}
  llvm.func @xsmm_brgemm_invoke(i64, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @xsmm_unary_dispatch(i64, i64, i64, i64, i64, i64, i64) -> i64 attributes {sym_visibility = "private"}
  llvm.func @xsmm_binary_dispatch(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 attributes {sym_visibility = "private"}
  llvm.func @xsmm_brgemm_dispatch(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 attributes {sym_visibility = "private"}
  llvm.func @xsmm_unary_scalar_invoke(i64, i64, f32, !llvm.ptr<f32>, i64) attributes {sym_visibility = "private"}
  llvm.func @MLP(%arg0: i64, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: i64, %arg5: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr)> 
    %3 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %4 = llvm.insertvalue %arg2, %3[0] : !llvm.struct<(i64, ptr)> 
    %5 = llvm.insertvalue %arg3, %4[1] : !llvm.struct<(i64, ptr)> 
    %6 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %7 = llvm.insertvalue %arg4, %6[0] : !llvm.struct<(i64, ptr)> 
    %8 = llvm.insertvalue %arg5, %7[1] : !llvm.struct<(i64, ptr)> 
    %9 = llvm.mlir.constant(4096 : index) : i64
    %10 = llvm.mlir.constant(1024 : index) : i64
    %11 = llvm.mlir.constant(32 : index) : i64
    %12 = llvm.mlir.constant(8388608 : index) : i64
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(128 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(262144 : index) : i64
    %17 = llvm.mlir.constant(4 : index) : i64
    %18 = llvm.mlir.constant(8388608 : index) : i64
    %19 = llvm.mlir.constant(1024 : index) : i64
    %20 = llvm.mlir.constant(32 : index) : i64
    %21 = llvm.mlir.constant(2 : index) : i64
    %22 = llvm.mlir.constant(8192 : index) : i64
    %23 = llvm.mlir.constant(0 : index) : i64
    %24 = llvm.mlir.constant(64 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(128 : index) : i64
    %27 = llvm.mlir.constant(262144 : index) : i64
    %28 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %29 = llvm.mlir.constant(8192 : i64) : i64
    %30 = llvm.mlir.constant(4 : index) : i64
    %31 = llvm.mlir.constant(8192 : index) : i64
    %32 = llvm.mlir.constant(2 : index) : i64
    %33 = llvm.mlir.constant(5 : i64) : i64
    %34 = llvm.mlir.constant(4 : i64) : i64
    %35 = llvm.mlir.constant(1024 : i64) : i64
    %36 = llvm.mlir.constant(0 : i64) : i64
    %37 = llvm.mlir.constant(262144 : i64) : i64
    %38 = llvm.mlir.constant(32 : i64) : i64
    %39 = llvm.mlir.constant(8 : i64) : i64
    %40 = llvm.mlir.constant(128 : i64) : i64
    %41 = llvm.mlir.constant(64 : i64) : i64
    %42 = llvm.mlir.constant(1 : i64) : i64
    %43 = llvm.mlir.constant(2 : i64) : i64
    %44 = llvm.extractvalue %2[1] : !llvm.struct<(i64, ptr)> 
    %45 = llvm.load %44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.extractvalue %5[1] : !llvm.struct<(i64, ptr)> 
    %47 = llvm.load %46 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %48 = llvm.extractvalue %8[1] : !llvm.struct<(i64, ptr)> 
    %49 = llvm.load %48 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %50 = llvm.extractvalue %49[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %51 = llvm.extractvalue %49[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %52 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %53 = llvm.insertvalue %50, %52[0] : !llvm.struct<(ptr, ptr, i64)> 
    %54 = llvm.insertvalue %51, %53[1] : !llvm.struct<(ptr, ptr, i64)> 
    %55 = llvm.mlir.constant(0 : index) : i64
    %56 = llvm.insertvalue %55, %54[2] : !llvm.struct<(ptr, ptr, i64)> 
    %57 = llvm.extractvalue %49[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %58 = llvm.extractvalue %49[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %59 = llvm.extractvalue %49[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %60 = llvm.extractvalue %49[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %61 = llvm.extractvalue %49[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %62 = llvm.extractvalue %49[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %63 = llvm.extractvalue %49[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %64 = llvm.mlir.zero : !llvm.ptr
    %65 = llvm.getelementptr %64[33554432] : (!llvm.ptr) -> !llvm.ptr, f32
    %66 = llvm.ptrtoint %65 : !llvm.ptr to i64
    %67 = llvm.add %66, %24  : i64
    %68 = llvm.call @malloc(%67) : (i64) -> !llvm.ptr
    %69 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %70 = llvm.sub %24, %25  : i64
    %71 = llvm.add %69, %70  : i64
    %72 = llvm.urem %71, %24  : i64
    %73 = llvm.sub %71, %72  : i64
    %74 = llvm.inttoptr %73 : i64 to !llvm.ptr
    %75 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %76 = llvm.insertvalue %68, %75[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.insertvalue %74, %76[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.insertvalue %23, %77[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.insertvalue %27, %78[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.insertvalue %26, %79[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.insertvalue %26, %80[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.insertvalue %25, %81[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%13 : i64)
  ^bb1(%83: i64):  // 2 preds: ^bb0, ^bb5
    %84 = llvm.icmp "slt" %83, %14 : i64
    llvm.cond_br %84, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%13 : i64)
  ^bb3(%85: i64):  // 2 preds: ^bb2, ^bb4
    %86 = llvm.icmp "slt" %85, %16 : i64
    llvm.cond_br %86, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %87 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.mlir.constant(262144 : index) : i64
    %89 = llvm.mul %83, %88  : i64
    %90 = llvm.add %89, %85  : i64
    %91 = llvm.getelementptr %87[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %92 = llvm.load %91 : !llvm.ptr -> f32
    %93 = llvm.mlir.constant(128 : index) : i64
    %94 = llvm.mul %85, %93  : i64
    %95 = llvm.add %94, %83  : i64
    %96 = llvm.getelementptr %74[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %92, %96 : f32, !llvm.ptr
    %97 = llvm.add %85, %15  : i64
    llvm.br ^bb3(%97 : i64)
  ^bb5:  // pred: ^bb3
    %98 = llvm.add %83, %15  : i64
    llvm.br ^bb1(%98 : i64)
  ^bb6:  // pred: ^bb1
    %99 = llvm.mlir.zero : !llvm.ptr
    %100 = llvm.getelementptr %99[8192] : (!llvm.ptr) -> !llvm.ptr, f32
    %101 = llvm.ptrtoint %100 : !llvm.ptr to i64
    %102 = llvm.add %101, %24  : i64
    %103 = llvm.call @malloc(%102) : (i64) -> !llvm.ptr
    %104 = llvm.ptrtoint %103 : !llvm.ptr to i64
    %105 = llvm.sub %24, %25  : i64
    %106 = llvm.add %104, %105  : i64
    %107 = llvm.urem %106, %24  : i64
    %108 = llvm.sub %106, %107  : i64
    %109 = llvm.inttoptr %108 : i64 to !llvm.ptr
    %110 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %111 = llvm.insertvalue %103, %110[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %109, %111[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %23, %112[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.insertvalue %24, %113[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.insertvalue %26, %114[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.insertvalue %26, %115[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.insertvalue %25, %116[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.call @xsmm_unary_dispatch(%43, %42, %41, %40, %42, %40, %39) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    %119 = llvm.ptrtoint %109 : !llvm.ptr to i64
    %120 = llvm.inttoptr %119 : i64 to !llvm.ptr<f32>
    llvm.call @xsmm_unary_scalar_invoke(%42, %118, %28, %120, %13) : (i64, i64, f32, !llvm.ptr<f32>, i64) -> ()
    %121 = llvm.mlir.zero : !llvm.ptr
    %122 = llvm.getelementptr %121[16777216] : (!llvm.ptr) -> !llvm.ptr, f32
    %123 = llvm.ptrtoint %122 : !llvm.ptr to i64
    %124 = llvm.add %123, %24  : i64
    %125 = llvm.call @malloc(%124) : (i64) -> !llvm.ptr
    %126 = llvm.ptrtoint %125 : !llvm.ptr to i64
    %127 = llvm.sub %24, %25  : i64
    %128 = llvm.add %126, %127  : i64
    %129 = llvm.urem %128, %24  : i64
    %130 = llvm.sub %128, %129  : i64
    %131 = llvm.inttoptr %130 : i64 to !llvm.ptr
    %132 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %133 = llvm.insertvalue %125, %132[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %134 = llvm.insertvalue %131, %133[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %135 = llvm.insertvalue %23, %134[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %136 = llvm.insertvalue %21, %135[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %137 = llvm.insertvalue %22, %136[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %138 = llvm.insertvalue %20, %137[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %139 = llvm.insertvalue %20, %138[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %140 = llvm.insertvalue %18, %139[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %141 = llvm.insertvalue %19, %140[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %142 = llvm.insertvalue %20, %141[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %143 = llvm.insertvalue %25, %142[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %144 = llvm.call @xsmm_unary_dispatch(%42, %42, %38, %38, %37, %38, %36) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    omp.parallel {
      omp.wsloop for  (%arg6, %arg7) : i64 = (%13, %13) to (%32, %31) step (%15, %15) {
        %186 = llvm.intr.stacksave : !llvm.ptr
        llvm.br ^bb1
      ^bb1:  // pred: ^bb0
        %187 = llvm.mul %arg6, %12  : i64
        %188 = llvm.mul %arg7, %11  : i64
        %189 = llvm.add %187, %188  : i64
        %190 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %191 = llvm.insertvalue %50, %190[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %192 = llvm.insertvalue %51, %191[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %193 = llvm.insertvalue %189, %192[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %194 = llvm.mlir.constant(32 : index) : i64
        %195 = llvm.insertvalue %194, %193[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %196 = llvm.mlir.constant(262144 : index) : i64
        %197 = llvm.insertvalue %196, %195[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %198 = llvm.mlir.constant(32 : index) : i64
        %199 = llvm.insertvalue %198, %197[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %200 = llvm.mlir.constant(1 : index) : i64
        %201 = llvm.insertvalue %200, %199[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %202 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
        %203 = llvm.insertvalue %125, %202[0] : !llvm.struct<(ptr, ptr, i64)> 
        %204 = llvm.insertvalue %131, %203[1] : !llvm.struct<(ptr, ptr, i64)> 
        %205 = llvm.mlir.constant(0 : index) : i64
        %206 = llvm.insertvalue %205, %204[2] : !llvm.struct<(ptr, ptr, i64)> 
        %207 = llvm.mul %arg6, %12  : i64
        %208 = llvm.mul %arg7, %10  : i64
        %209 = llvm.add %207, %208  : i64
        %210 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %211 = llvm.insertvalue %125, %210[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %212 = llvm.insertvalue %131, %211[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %213 = llvm.insertvalue %209, %212[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %214 = llvm.mlir.constant(32 : index) : i64
        %215 = llvm.insertvalue %214, %213[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %216 = llvm.mlir.constant(32 : index) : i64
        %217 = llvm.insertvalue %216, %215[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %218 = llvm.mlir.constant(32 : index) : i64
        %219 = llvm.insertvalue %218, %217[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %220 = llvm.mlir.constant(1 : index) : i64
        %221 = llvm.insertvalue %220, %219[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %222 = llvm.extractvalue %201[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %223 = llvm.extractvalue %201[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %224 = llvm.ptrtoint %223 : !llvm.ptr to i64
        %225 = llvm.inttoptr %224 : i64 to !llvm.ptr<f32>
        %226 = llvm.extractvalue %221[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %227 = llvm.extractvalue %221[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %228 = llvm.ptrtoint %227 : !llvm.ptr to i64
        %229 = llvm.inttoptr %228 : i64 to !llvm.ptr<f32>
        llvm.call @xsmm_unary_invoke(%42, %144, %225, %222, %229, %226) : (i64, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64) -> ()
        llvm.intr.stackrestore %186 : !llvm.ptr
        llvm.br ^bb2
      ^bb2:  // pred: ^bb1
        omp.yield
      }
      omp.terminator
    }
    %145 = llvm.mlir.zero : !llvm.ptr
    %146 = llvm.getelementptr %145[33554432] : (!llvm.ptr) -> !llvm.ptr, f32
    %147 = llvm.ptrtoint %146 : !llvm.ptr to i64
    %148 = llvm.add %147, %24  : i64
    %149 = llvm.call @malloc(%148) : (i64) -> !llvm.ptr
    %150 = llvm.ptrtoint %149 : !llvm.ptr to i64
    %151 = llvm.sub %24, %25  : i64
    %152 = llvm.add %150, %151  : i64
    %153 = llvm.urem %152, %24  : i64
    %154 = llvm.sub %152, %153  : i64
    %155 = llvm.inttoptr %154 : i64 to !llvm.ptr
    %156 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %157 = llvm.insertvalue %149, %156[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %158 = llvm.insertvalue %155, %157[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %159 = llvm.insertvalue %23, %158[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %160 = llvm.insertvalue %17, %159[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %161 = llvm.insertvalue %22, %160[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %162 = llvm.insertvalue %20, %161[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %163 = llvm.insertvalue %20, %162[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %164 = llvm.insertvalue %18, %163[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %165 = llvm.insertvalue %19, %164[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %166 = llvm.insertvalue %20, %165[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %167 = llvm.insertvalue %25, %166[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %168 = llvm.call @xsmm_unary_dispatch(%42, %42, %38, %38, %40, %38, %36) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    omp.parallel {
      omp.wsloop for  (%arg6, %arg7) : i64 = (%13, %13) to (%30, %31) step (%15, %15) {
        %186 = llvm.intr.stacksave : !llvm.ptr
        llvm.br ^bb1
      ^bb1:  // pred: ^bb0
        %187 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
        %188 = llvm.insertvalue %68, %187[0] : !llvm.struct<(ptr, ptr, i64)> 
        %189 = llvm.insertvalue %74, %188[1] : !llvm.struct<(ptr, ptr, i64)> 
        %190 = llvm.mlir.constant(0 : index) : i64
        %191 = llvm.insertvalue %190, %189[2] : !llvm.struct<(ptr, ptr, i64)> 
        %192 = llvm.mul %arg7, %9  : i64
        %193 = llvm.mul %arg6, %11  : i64
        %194 = llvm.add %192, %193  : i64
        %195 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %196 = llvm.insertvalue %68, %195[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %197 = llvm.insertvalue %74, %196[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %198 = llvm.insertvalue %194, %197[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %199 = llvm.mlir.constant(32 : index) : i64
        %200 = llvm.insertvalue %199, %198[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %201 = llvm.mlir.constant(128 : index) : i64
        %202 = llvm.insertvalue %201, %200[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %203 = llvm.mlir.constant(32 : index) : i64
        %204 = llvm.insertvalue %203, %202[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %205 = llvm.mlir.constant(1 : index) : i64
        %206 = llvm.insertvalue %205, %204[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %207 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
        %208 = llvm.insertvalue %149, %207[0] : !llvm.struct<(ptr, ptr, i64)> 
        %209 = llvm.insertvalue %155, %208[1] : !llvm.struct<(ptr, ptr, i64)> 
        %210 = llvm.mlir.constant(0 : index) : i64
        %211 = llvm.insertvalue %210, %209[2] : !llvm.struct<(ptr, ptr, i64)> 
        %212 = llvm.mul %arg6, %12  : i64
        %213 = llvm.mul %arg7, %10  : i64
        %214 = llvm.add %212, %213  : i64
        %215 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %216 = llvm.insertvalue %149, %215[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %217 = llvm.insertvalue %155, %216[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %218 = llvm.insertvalue %214, %217[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %219 = llvm.mlir.constant(32 : index) : i64
        %220 = llvm.insertvalue %219, %218[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %221 = llvm.mlir.constant(32 : index) : i64
        %222 = llvm.insertvalue %221, %220[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %223 = llvm.mlir.constant(32 : index) : i64
        %224 = llvm.insertvalue %223, %222[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %225 = llvm.mlir.constant(1 : index) : i64
        %226 = llvm.insertvalue %225, %224[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %227 = llvm.extractvalue %206[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %228 = llvm.extractvalue %206[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %229 = llvm.ptrtoint %228 : !llvm.ptr to i64
        %230 = llvm.inttoptr %229 : i64 to !llvm.ptr<f32>
        %231 = llvm.extractvalue %226[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %232 = llvm.extractvalue %226[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %233 = llvm.ptrtoint %232 : !llvm.ptr to i64
        %234 = llvm.inttoptr %233 : i64 to !llvm.ptr<f32>
        llvm.call @xsmm_unary_invoke(%42, %168, %230, %227, %234, %231) : (i64, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64) -> ()
        llvm.intr.stackrestore %186 : !llvm.ptr
        llvm.br ^bb2
      ^bb2:  // pred: ^bb1
        omp.yield
      }
      omp.terminator
    }
    %169 = llvm.extractvalue %47[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %170 = llvm.extractvalue %47[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %171 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %172 = llvm.insertvalue %169, %171[0] : !llvm.struct<(ptr, ptr, i64)> 
    %173 = llvm.insertvalue %170, %172[1] : !llvm.struct<(ptr, ptr, i64)> 
    %174 = llvm.mlir.constant(0 : index) : i64
    %175 = llvm.insertvalue %174, %173[2] : !llvm.struct<(ptr, ptr, i64)> 
    %176 = llvm.extractvalue %47[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %177 = llvm.extractvalue %47[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %178 = llvm.extractvalue %47[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %179 = llvm.call @xsmm_brgemm_dispatch(%42, %38, %38, %38, %38, %38, %38, %35, %35, %34) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    %180 = llvm.call @xsmm_binary_dispatch(%42, %42, %38, %38, %38, %38, %38, %34) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    %181 = llvm.call @xsmm_unary_dispatch(%33, %42, %38, %38, %38, %40, %36) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    omp.parallel {
      omp.wsloop for  (%arg6, %arg7) : i64 = (%13, %13) to (%32, %30) step (%15, %15) {
        %186 = llvm.intr.stacksave : !llvm.ptr
        llvm.br ^bb1
      ^bb1:  // pred: ^bb0
        %187 = llvm.mlir.zero : !llvm.ptr
        %188 = llvm.getelementptr %187[1024] : (!llvm.ptr) -> !llvm.ptr, f32
        %189 = llvm.ptrtoint %188 : !llvm.ptr to i64
        %190 = llvm.add %189, %24  : i64
        %191 = llvm.call @malloc(%190) : (i64) -> !llvm.ptr
        %192 = llvm.ptrtoint %191 : !llvm.ptr to i64
        %193 = llvm.sub %24, %25  : i64
        %194 = llvm.add %192, %193  : i64
        %195 = llvm.urem %194, %24  : i64
        %196 = llvm.sub %194, %195  : i64
        %197 = llvm.inttoptr %196 : i64 to !llvm.ptr
        %198 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
        %199 = llvm.insertvalue %125, %198[0] : !llvm.struct<(ptr, ptr, i64)> 
        %200 = llvm.insertvalue %131, %199[1] : !llvm.struct<(ptr, ptr, i64)> 
        %201 = llvm.mlir.constant(0 : index) : i64
        %202 = llvm.insertvalue %201, %200[2] : !llvm.struct<(ptr, ptr, i64)> 
        %203 = llvm.mul %arg6, %12  : i64
        %204 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
        %205 = llvm.insertvalue %125, %204[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %206 = llvm.insertvalue %131, %205[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %207 = llvm.insertvalue %203, %206[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %208 = llvm.mlir.constant(8192 : index) : i64
        %209 = llvm.insertvalue %208, %207[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %210 = llvm.mlir.constant(1024 : index) : i64
        %211 = llvm.insertvalue %210, %209[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %212 = llvm.mlir.constant(32 : index) : i64
        %213 = llvm.insertvalue %212, %211[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %214 = llvm.mlir.constant(32 : index) : i64
        %215 = llvm.insertvalue %214, %213[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %216 = llvm.mlir.constant(32 : index) : i64
        %217 = llvm.insertvalue %216, %215[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %218 = llvm.mlir.constant(1 : index) : i64
        %219 = llvm.insertvalue %218, %217[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %220 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
        %221 = llvm.insertvalue %149, %220[0] : !llvm.struct<(ptr, ptr, i64)> 
        %222 = llvm.insertvalue %155, %221[1] : !llvm.struct<(ptr, ptr, i64)> 
        %223 = llvm.mlir.constant(0 : index) : i64
        %224 = llvm.insertvalue %223, %222[2] : !llvm.struct<(ptr, ptr, i64)> 
        %225 = llvm.mul %arg7, %12  : i64
        %226 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
        %227 = llvm.insertvalue %149, %226[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %228 = llvm.insertvalue %155, %227[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %229 = llvm.insertvalue %225, %228[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %230 = llvm.mlir.constant(8192 : index) : i64
        %231 = llvm.insertvalue %230, %229[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %232 = llvm.mlir.constant(1024 : index) : i64
        %233 = llvm.insertvalue %232, %231[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %234 = llvm.mlir.constant(32 : index) : i64
        %235 = llvm.insertvalue %234, %233[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %236 = llvm.mlir.constant(32 : index) : i64
        %237 = llvm.insertvalue %236, %235[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %238 = llvm.mlir.constant(32 : index) : i64
        %239 = llvm.insertvalue %238, %237[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %240 = llvm.mlir.constant(1 : index) : i64
        %241 = llvm.insertvalue %240, %239[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %242 = llvm.extractvalue %219[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %243 = llvm.extractvalue %219[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %244 = llvm.ptrtoint %243 : !llvm.ptr to i64
        %245 = llvm.inttoptr %244 : i64 to !llvm.ptr<f32>
        %246 = llvm.extractvalue %241[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %247 = llvm.extractvalue %241[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
        %248 = llvm.ptrtoint %247 : !llvm.ptr to i64
        %249 = llvm.inttoptr %248 : i64 to !llvm.ptr<f32>
        %250 = llvm.ptrtoint %197 : !llvm.ptr to i64
        %251 = llvm.inttoptr %250 : i64 to !llvm.ptr<f32>
        llvm.call @xsmm_brgemm_invoke(%42, %179, %245, %242, %249, %246, %251, %13, %29) : (i64, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64, i64) -> ()
        %252 = llvm.mul %arg7, %11  : i64
        %253 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %254 = llvm.insertvalue %169, %253[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %255 = llvm.insertvalue %170, %254[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %256 = llvm.insertvalue %252, %255[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %257 = llvm.mlir.constant(32 : index) : i64
        %258 = llvm.insertvalue %257, %256[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %259 = llvm.mlir.constant(1 : index) : i64
        %260 = llvm.insertvalue %259, %258[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %261 = llvm.extractvalue %260[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %262 = llvm.extractvalue %260[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %263 = llvm.ptrtoint %262 : !llvm.ptr to i64
        %264 = llvm.inttoptr %263 : i64 to !llvm.ptr<f32>
        llvm.call @xsmm_binary_invoke(%42, %180, %264, %261, %251, %13, %251, %13) : (i64, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64) -> ()
        %265 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
        %266 = llvm.insertvalue %103, %265[0] : !llvm.struct<(ptr, ptr, i64)> 
        %267 = llvm.insertvalue %109, %266[1] : !llvm.struct<(ptr, ptr, i64)> 
        %268 = llvm.mlir.constant(0 : index) : i64
        %269 = llvm.insertvalue %268, %267[2] : !llvm.struct<(ptr, ptr, i64)> 
        %270 = llvm.mul %arg6, %9  : i64
        %271 = llvm.mul %arg7, %11  : i64
        %272 = llvm.add %270, %271  : i64
        %273 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %274 = llvm.insertvalue %103, %273[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %275 = llvm.insertvalue %109, %274[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %276 = llvm.insertvalue %272, %275[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %277 = llvm.mlir.constant(32 : index) : i64
        %278 = llvm.insertvalue %277, %276[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %279 = llvm.mlir.constant(128 : index) : i64
        %280 = llvm.insertvalue %279, %278[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %281 = llvm.mlir.constant(32 : index) : i64
        %282 = llvm.insertvalue %281, %280[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %283 = llvm.mlir.constant(1 : index) : i64
        %284 = llvm.insertvalue %283, %282[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %285 = llvm.extractvalue %284[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %286 = llvm.extractvalue %284[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %287 = llvm.ptrtoint %286 : !llvm.ptr to i64
        %288 = llvm.inttoptr %287 : i64 to !llvm.ptr<f32>
        llvm.call @xsmm_unary_invoke(%42, %181, %251, %13, %288, %285) : (i64, i64, !llvm.ptr<f32>, i64, !llvm.ptr<f32>, i64) -> ()
        llvm.call @free(%191) : (!llvm.ptr) -> ()
        llvm.intr.stackrestore %186 : !llvm.ptr
        llvm.br ^bb2
      ^bb2:  // pred: ^bb1
        omp.yield
      }
      omp.terminator
    }
    llvm.call @free(%68) : (!llvm.ptr) -> ()
    llvm.call @free(%125) : (!llvm.ptr) -> ()
    llvm.call @free(%149) : (!llvm.ptr) -> ()
    %182 = llvm.alloca %25 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %117, %182 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %183 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %184 = llvm.insertvalue %21, %183[0] : !llvm.struct<(i64, ptr)> 
    %185 = llvm.insertvalue %182, %184[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @refbackend_consume_func_return_mrf32(%21, %182) : (i64, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_MLP(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(i64, ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, ptr)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr)> 
    %3 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(i64, ptr)>
    %4 = llvm.extractvalue %3[0] : !llvm.struct<(i64, ptr)> 
    %5 = llvm.extractvalue %3[1] : !llvm.struct<(i64, ptr)> 
    %6 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(i64, ptr)>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(i64, ptr)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @MLP(%1, %2, %4, %5, %7, %8) : (i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr) -> ()
    llvm.return
  }
}

