add_rules("mode.debug", "mode.release")

add_includedirs("include")

option("cpu")
    set_default(true)
    set_showmenu(true)
    set_description("Enable or disable cpu kernel")
    add_defines("ENABLE_CPU")
option_end()

option("omp")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable OpenMP support for cpu kernel")
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

option("ascend-npu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Ascend NPU kernel")
    add_defines("ENABLE_ASCEND_NPU")
option_end()

option("iluvatar-bi")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Iluvatar Bi150 kernel")
    add_defines("ENABLE_ILU_BI")
option_end()

if is_mode("debug") then
    add_cxflags("-g -O0")
    add_defines("DEBUG_MODE")
end

if has_config("cpu") then

    add_defines("ENABLE_CPU")
    target("cpu")
        on_install(function (target) end)
        set_kind("static")

        if not is_plat("windows") then
            add_cxflags("-fPIC")
        end

        set_languages("cxx17")
        add_files("src/devices/cpu/*.cc", "src/ops/*/cpu/*.cc")
        if has_config("omp") then
            add_cxflags("-fopenmp")
            add_ldflags("-fopenmp")
        end
    target_end()

end

if has_config("nv-gpu") then

    add_defines("ENABLE_NV_GPU")
    target("nv-gpu")
        set_kind("static")
        on_install(function (target) end)
        set_policy("build.cuda.devlink", true)

        set_toolchains("cuda")
        add_links("cublas")
        add_links("cudnn")
        add_cugencodes("native")

        if is_plat("windows") then
            add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        else
            add_cuflags("-Xcompiler=-fPIC")
            add_culdflags("-Xcompiler=-fPIC")
            add_cxxflags("-fPIC")
        end

        set_languages("cxx17")
        add_files("src/devices/cuda/*.cc", "src/ops/*/cuda/*.cu")
        add_files("src/ops/*/cuda/*.cc")
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

        on_load(function (target)
            target:add("includedirs", "include")
        end)

        on_build_file(function (target, sourcefile)
            local objectfile = target:objectfile(sourcefile)
            os.mkdir(path.directory(objectfile))

            local cc = "/usr/local/neuware/bin/cncc"

            local includedirs = table.concat(target:get("includedirs"), " ")
            local args = {"-c", sourcefile, "-o", objectfile, "-I/usr/local/neuware/include", "--bang-mlu-arch=mtp_592", "-O3", "-fPIC", "-Wall", "-Werror", "-std=c++17", "-pthread"}
            
            for _, includedir in ipairs(target:get("includedirs")) do
                table.insert(args, "-I" .. includedir)
            end

            os.execv(cc, args)
            table.insert(target:objectfiles(), objectfile)
        end)

rule_end()


    target("cambricon-mlu")
        set_kind("static")
        on_install(function (target) end)
        set_languages("cxx17")
        add_files("src/devices/bang/*.cc", "src/ops/*/bang/*.cc")
        add_files("src/ops/*/bang/*.mlu", {rule = "mlu"})
        add_cxflags("-lstdc++ -Wall -Werror -fPIC")
    target_end()

end

if has_config("ascend-npu") then

    add_defines("ENABLE_ASCEND_NPU")
    local ASCEND_HOME = os.getenv("ASCEND_HOME")
    local SOC_VERSION = os.getenv("SOC_VERSION")

    -- Add include dirs
    add_includedirs(ASCEND_HOME .. "/include")
    add_includedirs(ASCEND_HOME .. "/include/aclnn")
    add_linkdirs(ASCEND_HOME .. "/lib64")
    add_links("libascendcl.so")
    add_links("libnnopbase.so")
    add_links("libopapi.so")
    add_links("libruntime.so")  
    add_linkdirs(ASCEND_HOME .. "/../../driver/lib64/driver")
    add_links("libascend_hal.so")
    local builddir = string.format(
            "%s/build/%s/%s/%s",
            os.projectdir(),
            get_config("plat"),
            get_config("arch"),
            get_config("mode")
        )
    rule("ascend-kernels")
        before_link(function ()
            local ascend_build_dir = path.join(os.projectdir(), "src/devices/ascend")
            os.cd(ascend_build_dir)
            os.exec("make")
            os.exec("cp $(projectdir)/src/devices/ascend/build/lib/libascend_kernels.a "..builddir.."/")
            os.cd(os.projectdir())
            
        end)
        after_clean(function ()
            local ascend_build_dir = path.join(os.projectdir(), "src/devices/ascend")
            os.cd(ascend_build_dir)
            os.exec("make clean")
            os.cd(os.projectdir())
            os.rm(builddir.. "/libascend_kernels.a")
            
        end)
        rule_end()

    target("ascend-npu")
        -- Other configs
        set_kind("static")
        set_languages("cxx17")
        on_install(function (target) end)
        -- Add files
        add_files("src/devices/ascend/*.cc", "src/ops/*/ascend/*.cc")
        add_cxflags("-lstdc++ -Wall -Werror -fPIC")

        -- Add operator 
        add_rules("ascend-kernels")
        add_links(builddir.."/libascend_kernels.a")

    target_end()
end


if has_config("iluvatar-bi") then

    add_defines("ENABLE_ILU_BI")
    -- Add include dirs
    add_includedirs("/usr/local/corex/include")
    add_includedirs("include")

    -- Add shared lib
    -- add_linkdirs("/usr/local/corex/lib")
    add_linkdirs("/usr/local/corex/lib64")
    add_links("cuda", "cublas", "cudnn", "cudart")
    add_ldflags("-Wl,--exclude-libs,libcudadevrt.a", {force = true})

    rule("cu")
        set_extensions(".cu")
        on_build_file(function (target, sourcefile)
            local objectfile = target:objectfile(sourcefile)
            os.mkdir(path.directory(objectfile))
            local mcc = "/usr/local/corex/bin/clang++"
            print("Compiling" .. sourcefile .. "to " .. objectfile)
            os.execv(mcc, {"-c", sourcefile, "-o", objectfile, "-I/usr/local/corex/include",  "-std=c++17", "-fPIC"})
            table.insert(target:objectfiles(), objectfile)
        end)
    rule_end()

    target("iluvatar-bi")
        set_kind("static")
        set_languages("cxx17")
        add_files("src/devices/iluvatar/*.cc", "src/ops/*/iluvatar/*.cc")
        add_files("src/ops/*/iluvatar/*.cu", {rule = "cu"})
        add_cxflags("-lstdc++ -Wall -Werror -fPIC")
    target_end()

end


target("infiniop")
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
    if has_config("ascend-npu") then
        add_deps("ascend-npu")
    end
    if has_config("iluvatar-bi") then
        add_deps("iluvatar-bi")
    end
    set_languages("cxx17")
    add_files("src/devices/handle.cc")
    add_files("src/ops/*/operator.cc")
    add_files("src/tensor/*.cc")

    after_build(function (target) 
        local builddir = string.format(
            "%s/build/%s/%s/%s",
            os.projectdir(),
            get_config("plat"),
            get_config("arch"),
            get_config("mode")
        )

        os.exec("mkdir -p $(projectdir)/lib/")
        os.exec("cp " ..builddir.. "/libinfiniop.so $(projectdir)/lib/")
        os.exec("cp -r $(projectdir)/include $(projectdir)/lib/")
        -- Define color codes
        local GREEN = '\27[0;32m'
        local YELLOW = '\27[1;33m'
        local NC = '\27[0m'  -- No Color

        -- Get the current directory
        local current_dir = os.curdir()

        -- Output messages with colors
        os.exec("echo -e '" .. GREEN .. "Compilation completed successfully." .. NC .. "'")
        os.exec("echo -e '" .. YELLOW .. "Install the libraries with \"xmake install\" or set INFINI_ROOT=" .. current_dir .. NC .. "'")
    end)
    
    on_install(function (target) 
        local home_dir = os.getenv("HOME")
        local infini_dir = home_dir .. "/.infini/"

        if os.isdir(infini_dir) then
            print("~/.infini/ detected, duplicated contents will be overwritten.")
        else
            os.mkdir(infini_dir)
        end
        os.exec("cp -r " .. "$(projectdir)/lib " .. infini_dir)

        local GREEN = '\27[0;32m'
        local YELLOW = '\27[1;33m'
        local NC = '\27[0m'  -- No Color
        os.exec("echo -e '" .. GREEN .. "Installation completed successfully at ~/.infini/." .. NC .. "'")
        os.exec("echo -e '" .. YELLOW .. "To set the environment variables, please run the following command:" .. NC .. "'")
        os.exec("echo -e '" .. YELLOW .. "echo \"export INFINI_ROOT=~/.infini/\" >> ~/.bashrc" .. NC .. "'")
        os.exec("echo -e '" .. YELLOW .. "echo \"export LD_LIBRARY_PATH=:~/.infini/lib:$LD_LIBRARY_PATH\" >> ~/.bashrc" .. NC .. "'")
    end)

target_end()
