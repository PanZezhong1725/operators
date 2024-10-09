add_rules("mode.debug", "mode.release")

add_includedirs("include")

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

option("ascend-npu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Ascend NPU kernel")
    add_defines("ENABLE_ASCEND_NPU")
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
        -- add_cxflags("-fopenmp")
        -- add_ldflags("-fopenmp")
    target_end()

end

if has_config("nv-gpu") then

    add_defines("ENABLE_NV_GPU")
    target("nv-gpu")
        set_kind("static")
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
    
    
    target("ascend-npu")
        -- Other configs
        set_kind("static")
        set_languages("cxx17")
        -- Add files
        add_files("src/devices/ascend/*.cc", "src/ops/*/ascend/*.cc")
        
        -- Add operator 
        add_linkdirs("src/ops/swiglu/ascend/build/lib")
        add_linkdirs("src/ops/rotary_embedding/ascend/build/lib")
        add_links("libswiglu.so")
        add_links("librope.so")

        add_rpathdirs("src/ops/swiglu/ascend/build/lib")
        add_rpathdirs("src/ops/rotary_embedding/ascend/build/lib")
        
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
    if has_config("ascend-npu") then
        add_deps("ascend-npu")
    end

    set_languages("cxx17")
    add_files("src/devices/handle.cc")
    add_files("src/ops/*/operator.cc")
    add_files("src/tensor/*.cc")
target_end()

task("install-operators")
    set_menu {
        usage = "xmake install-operators",
        description = "Build and install the operators",
        options = {}
    }
    on_run(function ()
        os.exec("xmake --root")
        os.exec("mkdir -p $(projectdir)/lib/")
        os.exec("cp $(projectdir)/build/linux/x86_64/release/liboperators.so $(projectdir)/lib/")
        os.exec("cp -r $(projectdir)/include $(projectdir)/lib/")
        -- Define color codes
        local GREEN = '\27[0;32m'
        local YELLOW = '\27[1;33m'
        local NC = '\27[0m'  -- No Color

        -- Get the current directory
        local current_dir = os.curdir()

        -- Output messages with colors
        os.exec("echo -e '" .. GREEN .. "Compilation completed successfully." .. NC .. "'")
        os.exec("echo -e '" .. YELLOW .. "To set the environment variable, please run the following command:" .. NC .. "'")
        os.exec("echo -e '" .. YELLOW .. "echo \"export INFINI_ROOT=" .. current_dir .. "/lib\" >> ~/.bashrc" .. NC .. "'")

    end)
