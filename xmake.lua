add_rules("mode.debug", "mode.release")

option("cpu")
    set_default(true)
    set_showmenu(true)
    set_description("Enable or disable cpu kernel")
    add_defines("ENABLE_CPU")
option_end()

option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Nvidia GPU kernel")
    add_defines("ENABLE_NV_GPU")
option_end()

option("cambricon-mlu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Cambricon MLU kernel")
    add_defines("ENABLE_CAMBRICON_MLU")
option_end()

if is_mode("debug") then
    add_cxflags("-g -O0")
    add_defines("DEBUG_MODE")
end

if has_config("cpu") then

    add_defines("ENABLE_CPU")
    target("cpu")
        set_kind("static")

        if not is_plat("windows") then
            add_cxflags("-fPIC")
        end

        set_languages("cxx17")
        add_files("src/devices/cpu/*.cc", "src/ops/*/cpu/*.cc")
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    target_end()

end

if has_config("nv-gpu") then

    add_defines("ENABLE_NV_GPU")
    target("nv-gpu")
        set_kind("static")
        set_policy("build.cuda.devlink", true)

        set_toolchains("cuda")
        add_links("cublas")
        add_cugencodes("native")

        if is_plat("windows") then
            add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        else
            add_cuflags("-Xcompiler=-fPIC")
            add_culdflags("-Xcompiler=-fPIC")
        end

        set_languages("cxx17")
        add_files("src/devices/cuda/*.cc", "src/ops/*/cuda/*.cu")
    target_end()

end

if has_config("cambricon-mlu") then

    add_defines("ENABLE_CAMBRICON_MLU")
    add_includedirs("/usr/local/neuware/include")
    add_linkdirs("/usr/local/neuware/lib64")
    add_linkdirs("/usr/local/neuware/lib")
    add_links("libcnrt.so")
    add_links("libcnnl.so")
    add_links("libcnnl_extra.so")
    add_links("libcnpapi.so")

    rule("mlu")
        set_extensions(".mlu")
        on_build_file(function (target, sourcefile)
            local objectfile = target:objectfile(sourcefile)
            os.mkdir(path.directory(objectfile))        
            local cc = "/usr/local/neuware/bin/cncc"
            print("Compiling " .. sourcefile .. " to " .. objectfile)
            os.execv(cc, {"-c", sourcefile, "-o", objectfile, "-I/usr/local/neuware/include", "--bang-mlu-arch=mtp_372", "-O3", "-fPIC", "-Wall", "-Werror", "-std=c++17", "-pthread"})
            table.insert(target:objectfiles(), objectfile)
        end)
    rule_end()

    target("cambricon-mlu")
        set_kind("static")
        set_languages("cxx17")
        add_files("src/devices/bang/*.cc", "src/ops/*/bang/*.cc")
        add_files("src/ops/*/bang/*.mlu", {rule = "mlu"})
        add_cxflags("-lstdc++ -Wall -Werror -fPIC")
    target_end()

end

target("operators")
    set_kind("shared")

    if has_config("cpu") then
        add_deps("cpu")
    end
    if has_config("nv-gpu") then
        add_deps("nv-gpu")
    end
    if has_config("cambricon-mlu") then
        add_deps("cambricon-mlu")
    end
    set_languages("cxx17")
    add_files("src/ops/*/operator.cc")
    add_files("src/tensor/*.cc")
target_end()

target("main")
    set_kind("binary")
    add_deps("operators")

    set_languages("c11")
    add_files("src/main.c")
target_end()
