  %1 = alloca i32, align 4
  %b = alloca [1 x %struct.float4], align 16
  %c = alloca [1 x %struct.float4], align 16
  %d = alloca [1 x %struct.float4], align 16
  %e = alloca [1 x %struct.float4], align 16
  store i32 0, i32* %1, align 4
  %2 = bitcast [1 x %struct.float4]* %b to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* bitcast ([1 x %struct.float4]* @main.b to i8*), i64 16, i32 16, i1 false)
  %5 = getelementptr inbounds [1 x %struct.float4], [1 x %struct.float4]* %b, i32 0, i32 0
  %6 = getelementptr inbounds [1 x %struct.float4], [1 x %struct.float4]* %c, i32 0, i32 0
  %7 = getelementptr inbounds [1 x %struct.float4], [1 x %struct.float4]* %d, i32 0, i32 0
  %8 = getelementptr inbounds [1 x %struct.float4], [1 x %struct.float4]* %e, i32 0, i32 0
  call void @A(i32 10, %struct.float4* %5, %struct.float4* %6, %struct.float4* %7, %struct.float4* %8)
store i32 10, i32* %a, align 8
store  %struct.float4* %3, %struct.float4** %b, align 8
store  %struct.float4* %4, %struct.float4** %c, align 8
store  %struct.float4* %5, %struct.float4** %d, align 8
store  %struct.float4* %6, %struct.float4** %e, align 8
  %1 = alloca i32, align 4
  %2 = alloca %struct.float4*, align 8
  %3 = alloca %struct.float4*, align 8
  %4 = alloca %struct.float4*, align 8
  %5 = alloca %struct.float4*, align 8
  %f = alloca i32, align 4
  %g = alloca i32, align 4
  %h = alloca %struct.float4, align 4
  %i = alloca %struct.float4, align 4
  %l = alloca %struct.float4, align 4
  %m = alloca %struct.float4, align 4
  %n = alloca %struct.float4, align 4
  store i32 %a, i32* %1, align 4
  store %struct.float4* %b, %struct.float4** %2, align 8
  store %struct.float4* %c, %struct.float4** %3, align 8
  store %struct.float4* %d, %struct.float4** %4, align 8
  store %struct.float4* %e, %struct.float4** %5, align 8
  store i32 0, i32* %f, align 4
  store i32 0, i32* %g, align 4
  %6 = load i32, i32* %g, align 4
  %7 = zext i32 %6 to i64
  %8 = load %struct.float4*, %struct.float4** %2, align 8
  %9 = getelementptr inbounds %struct.float4, %struct.float4* %8, i64 %7
  %10 = bitcast %struct.float4* %h to i8*
  %11 = bitcast %struct.float4* %9 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %10, i8* %11, i64 16, i32 4, i1 false)
  %14 = getelementptr inbounds %struct.float4, %struct.float4* %i, i32 0, i32 0
  %15 = getelementptr inbounds %struct.float4, %struct.float4* %h, i32 0, i32 0
  %16 = load float, float* %15, align 4
  %17 = fsub float 1.000000e+00, %16
  %18 = fmul float %17, 5.000000e+00
  %19 = getelementptr inbounds %struct.float4, %struct.float4* %h, i32 0, i32 0
  %20 = load float, float* %19, align 4
  %21 = fmul float %20, 3.000000e+01
  %22 = fadd float %18, %21
  store float %22, float* %14, align 4
  %23 = getelementptr inbounds %struct.float4, %struct.float4* %i, i32 0, i32 1
  store float 0.000000e+00, float* %23, align 4
  %24 = getelementptr inbounds %struct.float4, %struct.float4* %i, i32 0, i32 2
  store float 0.000000e+00, float* %24, align 4
  %25 = getelementptr inbounds %struct.float4, %struct.float4* %i, i32 0, i32 3
  store float 0.000000e+00, float* %25, align 4
  %26 = getelementptr inbounds %struct.float4, %struct.float4* %l, i32 0, i32 0
  %27 = getelementptr inbounds %struct.float4, %struct.float4* %i, i32 0, i32 0
  %28 = load float, float* %27, align 4
  %29 = load i32, i32* %1, align 4
  %30 = sitofp i32 %29 to float
  %31 = fdiv float %28, %30
  store float %31, float* %26, align 4
  %32 = getelementptr inbounds %struct.float4, %struct.float4* %l, i32 0, i32 1
  store float 0.000000e+00, float* %32, align 4
  %33 = getelementptr inbounds %struct.float4, %struct.float4* %l, i32 0, i32 2
  store float 0.000000e+00, float* %33, align 4
  %34 = getelementptr inbounds %struct.float4, %struct.float4* %l, i32 0, i32 3
  store float 0.000000e+00, float* %34, align 4
  %35 = bitcast %struct.float4* %l to { <2 x float>, <2 x float> }*
  %36 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %35, i32 0, i32 0
  %37 = load <2 x float>, <2 x float>* %36, align 4
  %38 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %35, i32 0, i32 1
  %39 = load <2 x float>, <2 x float>* %38, align 4
  %40 = call { <2 x float>, <2 x float> } @exp_f4(<2 x float> %37, <2 x float> %39)
  %1 = alloca %struct.float4, align 4
  %a = alloca %struct.float4, align 4
  %result = alloca %struct.float4, align 4
  %2 = bitcast %struct.float4* %a to { <2 x float>, <2 x float> }*
  %3 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %2, i32 0, i32 0
  store <2 x float> %a.coerce0, <2 x float>* %3, align 4
  %4 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %2, i32 0, i32 1
  store <2 x float> %a.coerce1, <2 x float>* %4, align 4
  %5 = getelementptr inbounds %struct.float4, %struct.float4* %result, i32 0, i32 0
  %6 = getelementptr inbounds %struct.float4, %struct.float4* %a, i32 0, i32 0
  %7 = load float, float* %6, align 4
  %8 = call float @expf(float %7) #4
  store float %9, float* %5, align 4
  %11 = getelementptr inbounds %struct.float4, %struct.float4* %result, i32 0, i32 1
  %12 = getelementptr inbounds %struct.float4, %struct.float4* %a, i32 0, i32 1
  %13 = load float, float* %12, align 4
  %14 = call float @expf(float %13) #5
  store float %15, float* %11, align 4
  %17 = getelementptr inbounds %struct.float4, %struct.float4* %result, i32 0, i32 2
  %18 = getelementptr inbounds %struct.float4, %struct.float4* %a, i32 0, i32 2
  %19 = load float, float* %18, align 4
  %20 = call float @expf(float %19) #5
  store float %21, float* %17, align 4
  %23 = getelementptr inbounds %struct.float4, %struct.float4* %result, i32 0, i32 3
  %24 = getelementptr inbounds %struct.float4, %struct.float4* %a, i32 0, i32 3
  %25 = load float, float* %24, align 4
  %26 = call float @expf(float %25) #5
  store float %27, float* %23, align 4
  %29 = bitcast %struct.float4* %1 to i8*
  %30 = bitcast %struct.float4* %result to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %29, i8* %30, i64 16, i32 4, i1 false)
  %33 = bitcast %struct.float4* %1 to { <2 x float>, <2 x float> }*
  %34 = load { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %33, align 4
  %43 = bitcast %struct.float4* %m to { <2 x float>, <2 x float> }*
  %44 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %43, i32 0, i32 0
  %45 = extractvalue { <2 x float>, <2 x float> } %41, 0
  store <2 x float> %45, <2 x float>* %44, align 4
  %46 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %43, i32 0, i32 1
  %47 = extractvalue { <2 x float>, <2 x float> } %41, 1
  store <2 x float> %47, <2 x float>* %46, align 4
  %48 = bitcast %struct.float4* %m to { <2 x float>, <2 x float> }*
  %49 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %48, i32 0, i32 0
  %50 = load <2 x float>, <2 x float>* %49, align 4
  %51 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %48, i32 0, i32 1
  %52 = load <2 x float>, <2 x float>* %51, align 4
  %53 = call { <2 x float>, <2 x float> } @sqrt_f4(<2 x float> %50, <2 x float> %52)
  %1 = alloca %struct.float4, align 4
  %a = alloca %struct.float4, align 4
  %result = alloca %struct.float4, align 4
  %2 = bitcast %struct.float4* %a to { <2 x float>, <2 x float> }*
  %3 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %2, i32 0, i32 0
  store <2 x float> %a.coerce0, <2 x float>* %3, align 4
  %4 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %2, i32 0, i32 1
  store <2 x float> %a.coerce1, <2 x float>* %4, align 4
  %5 = getelementptr inbounds %struct.float4, %struct.float4* %result, i32 0, i32 0
  %6 = getelementptr inbounds %struct.float4, %struct.float4* %a, i32 0, i32 0
  %7 = load float, float* %6, align 4
  %8 = call float @sqrtf(float %7) #5
  store float %9, float* %5, align 4
  %11 = getelementptr inbounds %struct.float4, %struct.float4* %result, i32 0, i32 1
  %12 = getelementptr inbounds %struct.float4, %struct.float4* %a, i32 0, i32 1
  %13 = load float, float* %12, align 4
  %14 = call float @sqrtf(float %13) #5
  store float %15, float* %11, align 4
  %17 = getelementptr inbounds %struct.float4, %struct.float4* %result, i32 0, i32 2
  %18 = getelementptr inbounds %struct.float4, %struct.float4* %a, i32 0, i32 2
  %19 = load float, float* %18, align 4
  %20 = call float @sqrtf(float %19) #5
  store float %21, float* %17, align 4
  %23 = getelementptr inbounds %struct.float4, %struct.float4* %result, i32 0, i32 3
  %24 = getelementptr inbounds %struct.float4, %struct.float4* %a, i32 0, i32 3
  %25 = load float, float* %24, align 4
  %26 = call float @sqrtf(float %25) #5
  store float %27, float* %23, align 4
  %29 = bitcast %struct.float4* %1 to i8*
  %30 = bitcast %struct.float4* %result to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %29, i8* %30, i64 16, i32 4, i1 false)
  %33 = bitcast %struct.float4* %1 to { <2 x float>, <2 x float> }*
  %34 = load { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %33, align 4
  %56 = bitcast %struct.float4* %n to { <2 x float>, <2 x float> }*
  %57 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %56, i32 0, i32 0
  %58 = extractvalue { <2 x float>, <2 x float> } %54, 0
  store <2 x float> %58, <2 x float>* %57, align 4
  %59 = getelementptr inbounds { <2 x float>, <2 x float> }, { <2 x float>, <2 x float> }* %56, i32 0, i32 1
  %60 = extractvalue { <2 x float>, <2 x float> } %54, 1
  store <2 x float> %60, <2 x float>* %59, align 4
  %61 = load %struct.float4*, %struct.float4** %3, align 8
  %62 = getelementptr inbounds %struct.float4, %struct.float4* %61, i64 0
  %63 = bitcast %struct.float4* %62 to i8*
  %64 = bitcast %struct.float4* %n to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %63, i8* %64, i64 16, i32 4, i1 false)
  %11 = getelementptr inbounds [1 x %struct.float4], [1 x %struct.float4]* %c, i64 0, i64 0
  %12 = getelementptr inbounds %struct.float4, %struct.float4* %11, i32 0, i32 0
  %13 = load float, float* %12, align 16
  %14 = fpext float %13 to double
  %15 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i32 0, i32 0), double %14)
