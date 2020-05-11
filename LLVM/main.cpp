#include <iostream>

#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/MCJIT.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

using namespace llvm;

template <int x>
[[clang::jit]] void run() {
    std::cout << "I was compiled at runtime, x = " << x << "\n";
}

int llvm_test()
{
    LLVMLinkInMCJIT();
    InitializeNativeTarget();

    LLVMContext Context;

    // Create some module to put our function into it.
    std::unique_ptr<Module> Owner = std::make_unique<Module>("test", Context);
    Module* M = Owner.get();

    // Create the add1 function entry and insert this entry into module M.  The
    // function will have a return type of "int" and take an argument of "int".
    Function* Add1F =
        Function::Create(FunctionType::get(Type::getInt32Ty(Context),
            { Type::getInt32Ty(Context) }, false),
            Function::ExternalLinkage, "add1", M);

    // Add a basic block to the function. As before, it automatically inserts
    // because of the last argument.
    BasicBlock* BB = BasicBlock::Create(Context, "EntryBlock", Add1F);

    // Create a basic block builder with default parameters.  The builder will
    // automatically append instructions to the basic block `BB'.
    IRBuilder<> builder(BB);

    // Get pointers to the constant `1'.
    Value* One = builder.getInt32(1);

    // Get pointers to the integer argument of the add1 function...
    assert(Add1F->arg_begin() != Add1F->arg_end()); // Make sure there's an arg
    Argument* ArgX = &*Add1F->arg_begin();          // Get the arg
    ArgX->setName("AnArg");            // Give it a nice symbolic name for fun.

    // Create the add instruction, inserting it into the end of BB.
    Value* Add = builder.CreateAdd(One, ArgX);

    // Create the return instruction and add it to the basic block
    builder.CreateRet(Add);

    // Now, function add1 is ready.

    // Now we're going to create function `foo', which returns an int and takes no
    // arguments.
    Function* FooF =
        Function::Create(FunctionType::get(Type::getInt32Ty(Context), {}, false),
            Function::ExternalLinkage, "foo", M);

    // Add a basic block to the FooF function.
    BB = BasicBlock::Create(Context, "EntryBlock", FooF);

    // Tell the basic block builder to attach itself to the new basic block
    builder.SetInsertPoint(BB);

    // Get pointer to the constant `10'.
    Value* Ten = builder.getInt32(10);

    // Pass Ten to the call to Add1F
    CallInst* Add1CallRes = builder.CreateCall(Add1F, Ten);
    Add1CallRes->setTailCall(true);

    // Create the return instruction and add it to the basic block.
    builder.CreateRet(Add1CallRes);

    // Now we create the JIT.
    std::string ebError;
    ExecutionEngine* EE = EngineBuilder(std::move(Owner))
        .setEngineKind(EngineKind::JIT).setErrorStr(&ebError).create();
    if (EE)
    {
        outs() << "We just constructed this LLVM module:\n\n" << *M;
        outs() << "\n\nRunning foo: ";
        outs().flush();

        // Call the `foo' function with no arguments:
        std::vector<GenericValue> noargs;
        GenericValue gv = EE->runFunction(FooF, noargs);

        // Import result of execution:
        outs() << "Result: " << gv.IntVal << "\n";
        delete EE;
    }
    else
    {
        outs() << ebError;
    }

    llvm_shutdown();
    return 0;
}

extern void clang_test();

int main(void)
{
    clang_test();
}

#pragma comment( lib, "LLVM-C.lib" )
#pragma comment( lib, "LLVMAArch64AsmParser.lib" )
#pragma comment( lib, "LLVMAArch64CodeGen.lib" )
#pragma comment( lib, "LLVMAArch64Desc.lib" )
#pragma comment( lib, "LLVMAArch64Disassembler.lib" )
#pragma comment( lib, "LLVMAArch64Info.lib" )
#pragma comment( lib, "LLVMAArch64Utils.lib" )
#pragma comment( lib, "LLVMAggressiveInstCombine.lib" )
#pragma comment( lib, "LLVMAMDGPUAsmParser.lib" )
#pragma comment( lib, "LLVMAMDGPUCodeGen.lib" )
#pragma comment( lib, "LLVMAMDGPUDesc.lib" )
#pragma comment( lib, "LLVMAMDGPUDisassembler.lib" )
#pragma comment( lib, "LLVMAMDGPUInfo.lib" )
#pragma comment( lib, "LLVMAMDGPUUtils.lib" )
#pragma comment( lib, "LLVMAnalysis.lib" )
#pragma comment( lib, "LLVMARMAsmParser.lib" )
#pragma comment( lib, "LLVMARMCodeGen.lib" )
#pragma comment( lib, "LLVMARMDesc.lib" )
#pragma comment( lib, "LLVMARMDisassembler.lib" )
#pragma comment( lib, "LLVMARMInfo.lib" )
#pragma comment( lib, "LLVMARMUtils.lib" )
#pragma comment( lib, "LLVMAsmParser.lib" )
#pragma comment( lib, "LLVMAsmPrinter.lib" )
#pragma comment( lib, "LLVMAVRAsmParser.lib" )
#pragma comment( lib, "LLVMAVRCodeGen.lib" )
#pragma comment( lib, "LLVMAVRDesc.lib" )
#pragma comment( lib, "LLVMAVRDisassembler.lib" )
#pragma comment( lib, "LLVMAVRInfo.lib" )
#pragma comment( lib, "LLVMBinaryFormat.lib" )
#pragma comment( lib, "LLVMBitReader.lib" )
#pragma comment( lib, "LLVMBitstreamReader.lib" )
#pragma comment( lib, "LLVMBitWriter.lib" )
#pragma comment( lib, "LLVMBPFAsmParser.lib" )
#pragma comment( lib, "LLVMBPFCodeGen.lib" )
#pragma comment( lib, "LLVMBPFDesc.lib" )
#pragma comment( lib, "LLVMBPFDisassembler.lib" )
#pragma comment( lib, "LLVMBPFInfo.lib" )
#pragma comment( lib, "LLVMCFGuard.lib" )
#pragma comment( lib, "LLVMCodeGen.lib" )
#pragma comment( lib, "LLVMCore.lib" )
#pragma comment( lib, "LLVMCoroutines.lib" )
#pragma comment( lib, "LLVMCoverage.lib" )
#pragma comment( lib, "LLVMDebugInfoCodeView.lib" )
#pragma comment( lib, "LLVMDebugInfoDWARF.lib" )
#pragma comment( lib, "LLVMDebugInfoGSYM.lib" )
#pragma comment( lib, "LLVMDebugInfoMSF.lib" )
#pragma comment( lib, "LLVMDebugInfoPDB.lib" )
#pragma comment( lib, "LLVMDemangle.lib" )
#pragma comment( lib, "LLVMDlltoolDriver.lib" )
#pragma comment( lib, "LLVMDWARFLinker.lib" )
#pragma comment( lib, "LLVMExecutionEngine.lib" )
#pragma comment( lib, "LLVMExtensions.lib" )
#pragma comment( lib, "LLVMFrontendOpenMP.lib" )
#pragma comment( lib, "LLVMFuzzMutate.lib" )
#pragma comment( lib, "LLVMGlobalISel.lib" )
#pragma comment( lib, "LLVMHexagonAsmParser.lib" )
#pragma comment( lib, "LLVMHexagonCodeGen.lib" )
#pragma comment( lib, "LLVMHexagonDesc.lib" )
#pragma comment( lib, "LLVMHexagonDisassembler.lib" )
#pragma comment( lib, "LLVMHexagonInfo.lib" )
#pragma comment( lib, "LLVMInstCombine.lib" )
#pragma comment( lib, "LLVMInstrumentation.lib" )
#pragma comment( lib, "LLVMInterpreter.lib" )
#pragma comment( lib, "LLVMipo.lib" )
#pragma comment( lib, "LLVMIRReader.lib" )
#pragma comment( lib, "LLVMJITLink.lib" )
#pragma comment( lib, "LLVMLanaiAsmParser.lib" )
#pragma comment( lib, "LLVMLanaiCodeGen.lib" )
#pragma comment( lib, "LLVMLanaiDesc.lib" )
#pragma comment( lib, "LLVMLanaiDisassembler.lib" )
#pragma comment( lib, "LLVMLanaiInfo.lib" )
#pragma comment( lib, "LLVMLibDriver.lib" )
#pragma comment( lib, "LLVMLineEditor.lib" )
#pragma comment( lib, "LLVMLinker.lib" )
#pragma comment( lib, "LLVMLTO.lib" )
#pragma comment( lib, "LLVMMC.lib" )
#pragma comment( lib, "LLVMMCA.lib" )
#pragma comment( lib, "LLVMMCDisassembler.lib" )
#pragma comment( lib, "LLVMMCJIT.lib" )
#pragma comment( lib, "LLVMMCParser.lib" )
#pragma comment( lib, "LLVMMipsAsmParser.lib" )
#pragma comment( lib, "LLVMMipsCodeGen.lib" )
#pragma comment( lib, "LLVMMipsDesc.lib" )
#pragma comment( lib, "LLVMMipsDisassembler.lib" )
#pragma comment( lib, "LLVMMipsInfo.lib" )
#pragma comment( lib, "LLVMMIRParser.lib" )
#pragma comment( lib, "LLVMMSP430AsmParser.lib" )
#pragma comment( lib, "LLVMMSP430CodeGen.lib" )
#pragma comment( lib, "LLVMMSP430Desc.lib" )
#pragma comment( lib, "LLVMMSP430Disassembler.lib" )
#pragma comment( lib, "LLVMMSP430Info.lib" )
#pragma comment( lib, "LLVMNVPTXCodeGen.lib" )
#pragma comment( lib, "LLVMNVPTXDesc.lib" )
#pragma comment( lib, "LLVMNVPTXInfo.lib" )
#pragma comment( lib, "LLVMObjCARCOpts.lib" )
#pragma comment( lib, "LLVMObject.lib" )
#pragma comment( lib, "LLVMObjectYAML.lib" )
#pragma comment( lib, "LLVMOption.lib" )
#pragma comment( lib, "LLVMOrcError.lib" )
#pragma comment( lib, "LLVMOrcJIT.lib" )
#pragma comment( lib, "LLVMPasses.lib" )
#pragma comment( lib, "LLVMPowerPCAsmParser.lib" )
#pragma comment( lib, "LLVMPowerPCCodeGen.lib" )
#pragma comment( lib, "LLVMPowerPCDesc.lib" )
#pragma comment( lib, "LLVMPowerPCDisassembler.lib" )
#pragma comment( lib, "LLVMPowerPCInfo.lib" )
#pragma comment( lib, "LLVMProfileData.lib" )
#pragma comment( lib, "LLVMRemarks.lib" )
#pragma comment( lib, "LLVMRISCVAsmParser.lib" )
#pragma comment( lib, "LLVMRISCVCodeGen.lib" )
#pragma comment( lib, "LLVMRISCVDesc.lib" )
#pragma comment( lib, "LLVMRISCVDisassembler.lib" )
#pragma comment( lib, "LLVMRISCVInfo.lib" )
#pragma comment( lib, "LLVMRISCVUtils.lib" )
#pragma comment( lib, "LLVMRuntimeDyld.lib" )
#pragma comment( lib, "LLVMScalarOpts.lib" )
#pragma comment( lib, "LLVMSelectionDAG.lib" )
#pragma comment( lib, "LLVMSparcAsmParser.lib" )
#pragma comment( lib, "LLVMSparcCodeGen.lib" )
#pragma comment( lib, "LLVMSparcDesc.lib" )
#pragma comment( lib, "LLVMSparcDisassembler.lib" )
#pragma comment( lib, "LLVMSparcInfo.lib" )
#pragma comment( lib, "LLVMSupport.lib" )
#pragma comment( lib, "LLVMSymbolize.lib" )
#pragma comment( lib, "LLVMSystemZAsmParser.lib" )
#pragma comment( lib, "LLVMSystemZCodeGen.lib" )
#pragma comment( lib, "LLVMSystemZDesc.lib" )
#pragma comment( lib, "LLVMSystemZDisassembler.lib" )
#pragma comment( lib, "LLVMSystemZInfo.lib" )
#pragma comment( lib, "LLVMTableGen.lib" )
#pragma comment( lib, "LLVMTarget.lib" )
#pragma comment( lib, "LLVMTextAPI.lib" )
#pragma comment( lib, "LLVMTransformUtils.lib" )
#pragma comment( lib, "LLVMVectorize.lib" )
#pragma comment( lib, "LLVMWebAssemblyAsmParser.lib" )
#pragma comment( lib, "LLVMWebAssemblyCodeGen.lib" )
#pragma comment( lib, "LLVMWebAssemblyDesc.lib" )
#pragma comment( lib, "LLVMWebAssemblyDisassembler.lib" )
#pragma comment( lib, "LLVMWebAssemblyInfo.lib" )
#pragma comment( lib, "LLVMWindowsManifest.lib" )
#pragma comment( lib, "LLVMX86AsmParser.lib" )
#pragma comment( lib, "LLVMX86CodeGen.lib" )
#pragma comment( lib, "LLVMX86Desc.lib" )
#pragma comment( lib, "LLVMX86Disassembler.lib" )
#pragma comment( lib, "LLVMX86Info.lib" )
#pragma comment( lib, "LLVMXCoreCodeGen.lib" )
#pragma comment( lib, "LLVMXCoreDesc.lib" )
#pragma comment( lib, "LLVMXCoreDisassembler.lib" )
#pragma comment( lib, "LLVMXCoreInfo.lib" )
#pragma comment( lib, "LLVMXRay.lib" )
#pragma comment( lib, "LTO.lib" )
#pragma comment( lib, "Remarks.lib" )