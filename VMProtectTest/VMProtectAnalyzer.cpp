#include "pch.h"

#include "VMProtectAnalyzer.hpp"
#include "x86_instruction.hpp"
#include "AbstractStream.hpp"
#include "CFG.hpp"
#include "IR.hpp"

constexpr bool cout_vm_enter_instructions = 1;

// helper?
void print_basic_blocks(const std::shared_ptr<BasicBlock> &first_basic_block)
{
	std::set<unsigned long long> visit_for_print;
	std::shared_ptr<BasicBlock> basic_block = first_basic_block;
	for (auto it = basic_block->instructions.begin(); it != basic_block->instructions.end();)
	{
		const auto& instruction = *it;
		if (++it != basic_block->instructions.end())
		{
			// loop until it reaches end
			instruction->print();
			continue;
		}

		// dont print unconditional jmp, they are annoying
		if (instruction->get_category() != XED_CATEGORY_UNCOND_BR
			|| instruction->get_branch_displacement_width() == 0)
		{
			instruction->print();
		}

		visit_for_print.insert(basic_block->leader);
		if (basic_block->next_basic_block && visit_for_print.count(basic_block->next_basic_block->leader) <= 0)
		{
			// print next
			basic_block = basic_block->next_basic_block;
		}
		else if (basic_block->target_basic_block && visit_for_print.count(basic_block->target_basic_block->leader) <= 0)
		{
			// it ends with jmp?
			basic_block = basic_block->target_basic_block;
		}
		else
		{
			// perhaps finishes?
			break;
		}

		it = basic_block->instructions.begin();
	}
}

// variablenode?
triton::engines::symbolic::SharedSymbolicVariable get_symbolic_var(const triton::ast::SharedAbstractNode &node)
{
	return node->getType() == triton::ast::VARIABLE_NODE ? 
		std::dynamic_pointer_cast<triton::ast::VariableNode>(node)->getSymbolicVariable() : nullptr;
}
std::set<triton::ast::SharedAbstractNode> collect_symvars(const triton::ast::SharedAbstractNode &parent)
{
	std::set<triton::ast::SharedAbstractNode> result;
	if (!parent)
		return result;

	if (parent->getChildren().empty() && parent->isSymbolized())
	{
		// this must be variable node right?
		assert(parent->getType() == triton::ast::VARIABLE_NODE);
		result.insert(parent);
	}

	for (const triton::ast::SharedAbstractNode &child : parent->getChildren())
	{
		if (!child->getChildren().empty())
		{
			// go deep if symbolized
			if (child->isSymbolized())
			{
				auto _new = collect_symvars(child);
				result.insert(_new.begin(), _new.end());
			}
		}
		else if (child->isSymbolized())
		{
			// this must be variable node right?
			assert(child->getType() == triton::ast::VARIABLE_NODE);
			result.insert(child);
		}
	}
	return result;
}
bool is_unary_operation(const triton::arch::Instruction &triton_instruction)
{
	switch (triton_instruction.getType())
	{
		case triton::arch::x86::ID_INS_INC:
		case triton::arch::x86::ID_INS_DEC:
		case triton::arch::x86::ID_INS_NEG:
		case triton::arch::x86::ID_INS_NOT:
			return true;

		default:
			return false;
	}
}
bool is_binary_operation(const triton::arch::Instruction &triton_instruction)
{
	switch (triton_instruction.getType())
	{
		case triton::arch::x86::ID_INS_ADD:
		case triton::arch::x86::ID_INS_SUB:
		case triton::arch::x86::ID_INS_SHL:
		case triton::arch::x86::ID_INS_SHR:
		case triton::arch::x86::ID_INS_RCR:
		case triton::arch::x86::ID_INS_RCL:
		case triton::arch::x86::ID_INS_ROL:
		case triton::arch::x86::ID_INS_ROR:
		case triton::arch::x86::ID_INS_AND:
		case triton::arch::x86::ID_INS_OR:
		case triton::arch::x86::ID_INS_XOR:
		//case triton::arch::x86::ID_INS_CMP:
		//case triton::arch::x86::ID_INS_TEST:
		case triton::arch::x86::ID_INS_MUL:
		case triton::arch::x86::ID_INS_IMUL:
			return true;

		default:
			return false;
	}
}


// VMProtectAnalyzer
VMProtectAnalyzer::VMProtectAnalyzer(triton::arch::architecture_e arch)
{
	triton_api = std::make_shared<triton::API>();
	triton_api->setArchitecture(arch);
	triton_api->setMode(triton::modes::ALIGNED_MEMORY, true);
	triton_api->setMode(triton::modes::CONSTANT_FOLDING, true);
	//triton_api->setAstRepresentationMode(triton::ast::representations::PYTHON_REPRESENTATION);
	this->m_scratch_size = 0;
}
VMProtectAnalyzer::~VMProtectAnalyzer()
{
}

bool VMProtectAnalyzer::is_x64() const
{
	const triton::arch::architecture_e architecture = this->triton_api->getArchitecture();
	switch (architecture)
	{
		case triton::arch::ARCH_X86:
			return false;

		case triton::arch::ARCH_X86_64:
			return true;

		default:
			throw std::runtime_error("invalid architecture");
	}
}

triton::arch::Register VMProtectAnalyzer::get_bp_register() const
{
	return this->is_x64() ? triton_api->registers.x86_rbp : triton_api->registers.x86_ebp;
}
triton::arch::Register VMProtectAnalyzer::get_sp_register() const
{
	const triton::arch::CpuInterface *_cpu = triton_api->getCpuInstance();
	if (!_cpu)
		throw std::runtime_error("CpuInterface is nullptr");

	return _cpu->getStackPointer();
}
triton::arch::Register VMProtectAnalyzer::get_ip_register() const
{
	const triton::arch::CpuInterface *_cpu = triton_api->getCpuInstance();
	if (!_cpu)
		throw std::runtime_error("CpuInterface is nullptr");

	return _cpu->getProgramCounter();
}

triton::uint64 VMProtectAnalyzer::get_bp() const
{
	return triton_api->getConcreteRegisterValue(this->get_bp_register()).convert_to<triton::uint64>();
}
triton::uint64 VMProtectAnalyzer::get_sp() const
{
	return triton_api->getConcreteRegisterValue(this->get_sp_register()).convert_to<triton::uint64>();
}
triton::uint64 VMProtectAnalyzer::get_ip() const
{
	return triton_api->getConcreteRegisterValue(this->get_ip_register()).convert_to<triton::uint64>();
}

bool VMProtectAnalyzer::is_bytecode_address(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context)
{
	// return true if lea_ast is constructed by bytecode
	const std::set<triton::ast::SharedAbstractNode> symvars = collect_symvars(lea_ast);
	if (symvars.empty())
		return false;

	for (auto it = symvars.begin(); it != symvars.end(); ++it)
	{
		const triton::ast::SharedAbstractNode &node = *it;
		const triton::engines::symbolic::SharedSymbolicVariable &symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(node)->getSymbolicVariable();
		if (symvar->getId() != context->symvar_bytecode->getId())
			return false;
	}
	return true;
}
bool VMProtectAnalyzer::is_stack_address(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context)
{
	// return true if lea_ast is constructed by stack
	const std::set<triton::ast::SharedAbstractNode> symvars = collect_symvars(lea_ast);
	if (symvars.empty())
		return false;

	for (auto it = symvars.begin(); it != symvars.end(); ++it)
	{
		const triton::ast::SharedAbstractNode &node = *it;
		const triton::engines::symbolic::SharedSymbolicVariable &symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(node)->getSymbolicVariable();
		if (symvar != context->symvar_vmp_sp)
			return false;
	}
	return true;
}
bool VMProtectAnalyzer::is_scratch_area_address(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context)
{
	// size is hardcoded for now (can see in any push handler perhaps)
	const triton::uint64 runtime_address = lea_ast->evaluate().convert_to<triton::uint64>();
	return context->x86_sp <= runtime_address && runtime_address < (context->x86_sp + context->scratch_area_size);
}
bool VMProtectAnalyzer::is_fetch_arguments(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context)
{
	if (lea_ast->getType() != triton::ast::VARIABLE_NODE)
		return false;

	const triton::engines::symbolic::SharedSymbolicVariable &symvar =
		std::dynamic_pointer_cast<triton::ast::VariableNode>(lea_ast)->getSymbolicVariable();
	return context->arguments.find(symvar->getId()) != context->arguments.end();
}

void VMProtectAnalyzer::load(AbstractStream& stream,
	unsigned long long module_base, unsigned long long vmp0_address, unsigned long long vmp0_size)
{
	// concretize vmp section memory
	unsigned long long vmp_section_address = (module_base + vmp0_address);
	unsigned long long vmp_section_size = vmp0_size;
	void *vmp0 = malloc(vmp_section_size);

	stream.seek(vmp_section_address);
	if (stream.read(vmp0, vmp_section_size) != vmp_section_size)
		throw std::runtime_error("stream.read failed");

	triton_api->setConcreteMemoryAreaValue(vmp_section_address, (const triton::uint8 *)vmp0, vmp_section_size);
	free(vmp0);
}

// vm-enter
std::map<triton::usize, std::shared_ptr<IR::Register>> VMProtectAnalyzer::symbolize_registers()
{
	std::map<triton::usize, std::shared_ptr<IR::Register>> regmap;
	auto _work = [this, &regmap](const triton::arch::Register& reg)
	{
		auto symvar = triton_api->symbolizeRegister(reg);
		symvar->setAlias(reg.getName());
		regmap.insert(std::make_pair(symvar->getId(), std::make_shared<IR::Register>(reg)));
	};
	if (this->is_x64())
	{
		_work(triton_api->registers.x86_rax);
		_work(triton_api->registers.x86_rbx);
		_work(triton_api->registers.x86_rcx);
		_work(triton_api->registers.x86_rdx);
		_work(triton_api->registers.x86_rsi);
		_work(triton_api->registers.x86_rdi);
		_work(triton_api->registers.x86_rbp);
		_work(triton_api->registers.x86_r8);
		_work(triton_api->registers.x86_r9);
		_work(triton_api->registers.x86_r10);
		_work(triton_api->registers.x86_r11);
		_work(triton_api->registers.x86_r12);
		_work(triton_api->registers.x86_r13);
		_work(triton_api->registers.x86_r14);
		_work(triton_api->registers.x86_r15);
	}
	else
	{
		_work(triton_api->registers.x86_eax);
		_work(triton_api->registers.x86_ebx);
		_work(triton_api->registers.x86_ecx);
		_work(triton_api->registers.x86_edx);
		_work(triton_api->registers.x86_esi);
		_work(triton_api->registers.x86_edi);
		_work(triton_api->registers.x86_ebp);
	}
	return regmap;
}
void VMProtectAnalyzer::analyze_vm_enter(AbstractStream& stream, triton::uint64 address)
{
	// reset triton api
	triton_api->clearCallbacks();
	triton_api->concretizeAllMemory();
	triton_api->concretizeAllRegister();
	auto regmap = this->symbolize_registers();

	// set esp
	const triton::arch::Register sp_register = this->get_sp_register();
	triton_api->setConcreteRegisterValue(sp_register, 0x1000);

	const triton::uint64 previous_sp = this->get_sp();
	bool check_flags = true;

	std::shared_ptr<BasicBlock> basic_block = make_cfg(stream, address);
	for (auto it = basic_block->instructions.begin(); it != basic_block->instructions.end();)
	{
		const std::shared_ptr<x86_instruction> xed_instruction = *it;
		const std::vector<xed_uint8_t> bytes = xed_instruction->get_bytes();

		// concrete ip as some instruction read (E|R)IP
		triton_api->setConcreteRegisterValue(this->get_ip_register(), xed_instruction->get_addr());

		// do stuff with triton
		triton::arch::Instruction triton_instruction;
		triton_instruction.setOpcode(&bytes[0], (triton::uint32)bytes.size());
		triton_instruction.setAddress(xed_instruction->get_addr());
		if (!triton_api->processing(triton_instruction))
		{
			throw std::runtime_error("triton processing failed");
		}

		// check flags
		if (check_flags)
		{
			// symbolize memory if pushfd or pushfq
			if (triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFD
				|| triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFQ)
			{
				const auto& stores = triton_instruction.getStoreAccess();
				assert(stores.size() == 1);

				auto symvar_eflags = triton_api->symbolizeMemory(stores.begin()->first);
				regmap.insert(std::make_pair(symvar_eflags->getId(), std::make_shared<IR::Register>(triton_api->registers.x86_eflags)));
			}

			// written_register
			for (const auto &pair : triton_instruction.getWrittenRegisters())
			{
				const triton::arch::Register &written_register = pair.first;
				if (triton_api->isFlag(written_register))
				{
					check_flags = false;
					break;
				}
			}
		}

		if (++it != basic_block->instructions.end())
		{
			// loop until it reaches end
			if (cout_vm_enter_instructions)
				std::cout << triton_instruction << "\n";
			continue;
		}

		if (xed_instruction->get_category() != XED_CATEGORY_UNCOND_BR
			|| xed_instruction->get_branch_displacement_width() == 0)
		{
			if (cout_vm_enter_instructions)
				std::cout << triton_instruction << "\n";
		}

		if (basic_block->next_basic_block && basic_block->target_basic_block)
		{
			// it ends with conditional branch
			if (triton_instruction.isConditionTaken())
			{
				basic_block = basic_block->target_basic_block;
			}
			else
			{
				basic_block = basic_block->next_basic_block;
			}
		}
		else if (basic_block->target_basic_block)
		{
			// it ends with jmp?
			basic_block = basic_block->target_basic_block;
		}
		else if (basic_block->next_basic_block)
		{
			// just follow :)
			basic_block = basic_block->next_basic_block;
		}
		else
		{
			// perhaps finishes?
			assert(basic_block->terminator);
			break;
		}

		it = basic_block->instructions.begin();
	}

	// create instructions
	std::list<std::shared_ptr<IR::Instruction>> ir_instructions;
	const triton::uint64 bp = this->get_bp();
	const triton::uint64 sp = this->get_sp();
	const triton::uint64 scratch_size = bp - sp;
	const triton::uint64 scratch_length = scratch_size / triton_api->getGprSize();
	const triton::uint64 var_length = (previous_sp - bp) / triton_api->getGprSize();
	for (triton::uint64 i = 0; i < var_length; i++)
	{
		triton::ast::SharedAbstractNode mem_ast = triton_api->getMemoryAst(
			triton::arch::MemoryAccess(previous_sp - (i * triton_api->getGprSize()) - triton_api->getGprSize(), triton_api->getGprSize()));
		triton::ast::SharedAbstractNode simplified = triton_api->processSimplification(mem_ast, true);
		if (!simplified->isSymbolized())
		{
			// should be immediate if not symbolized
			const triton::uint64 imm = simplified->evaluate().convert_to<triton::uint64>();
			auto _push = std::make_shared<IR::Push>(std::make_shared<IR::Immediate>(imm));
			ir_instructions.push_back(std::move(_push));
		}
		else if (simplified->getType() == triton::ast::VARIABLE_NODE)
		{
			const auto symvar = get_symbolic_var(simplified);
			auto it = regmap.find(symvar->getId());
			if (it == regmap.end())
			{
				std::stringstream ss;
				ss << "L: " << __LINE__ << " vm enter error " << symvar;
				throw std::runtime_error(ss.str());
			}

			auto _push = std::make_shared<IR::Push>(it->second);
			ir_instructions.push_back(std::move(_push));
		}
		else
		{
			throw std::runtime_error("vm enter error");
		}
	}

	std::cout << "scratch_size: 0x" << std::hex << scratch_size << ", scratch_length: " << std::dec << scratch_length << "\n";
	this->m_scratch_size = scratch_size;
	this->m_vmp_instructions = std::move(ir_instructions);
}


// vm-handler
void VMProtectAnalyzer::symbolize_memory(const triton::arch::MemoryAccess& mem, VMPHandlerContext *context)
{
	const triton::uint64 mem_address = mem.getAddress();
	triton::ast::SharedAbstractNode lea_ast = mem.getLeaAst();
	if (!lea_ast)
	{
		// most likely can be ignored
		return;
	}

	lea_ast = triton_api->processSimplification(lea_ast, true);
	if (!lea_ast->isSymbolized())
	{
		// most likely can be ignored
		return;
	}

	if (this->is_bytecode_address(lea_ast, context))
	{
		// bytecode can be considered const value
		//triton_api->taintMemory(mem);
	}

	// lea_ast = context + const
	else if (this->is_scratch_area_address(lea_ast, context))
	{
		// [EBP+offset]
		const triton::uint64 scratch_offset = lea_ast->evaluate().convert_to<triton::uint64>() - context->x86_sp;

		triton::engines::symbolic::SharedSymbolicVariable symvar_vmreg = triton_api->symbolizeMemory(mem);
		context->scratch_variables.insert(std::make_pair(symvar_vmreg->getId(), symvar_vmreg));
		std::cout << "Load Scratch:[0x" << std::hex << scratch_offset << "]\n";

		// TempVar = VM_REG
		auto temp_variable = IR::Variable::create_variable(mem.getSize());

		auto ir_imm = std::make_shared<IR::Immediate>(scratch_offset);
		std::shared_ptr<IR::Expression> right_expression = std::make_shared<IR::Memory>(ir_imm, IR::ir_segment_scratch, (IR::ir_size)mem.getSize());

		context->instructions.push_back(std::make_shared<IR::Assign>(temp_variable, right_expression));
		context->expression_map[symvar_vmreg->getId()] = temp_variable;
		symvar_vmreg->setAlias(temp_variable->get_name());
	}
	else if (this->is_stack_address(lea_ast, context))
	{
		const triton::uint64 offset = mem_address - context->vmp_sp;
		triton::arch::Register segment_register = mem.getConstSegmentRegister();
		if (segment_register.getId() == triton::arch::ID_REG_INVALID)
		{
			// DS?
			//segment_register = triton_api->registers.x86_ds;
		}

		triton::engines::symbolic::SharedSymbolicVariable symvar_arg = triton_api->symbolizeMemory(mem);
		context->arguments.insert(std::make_pair(symvar_arg->getId(), symvar_arg));
		std::cout << "Load [vmp_sp+0x" << std::hex << offset << "]\n";

		// add(vmp_sp, offset)
		auto expr = std::make_shared<IR::Add>(context->expression_map[context->symvar_vmp_sp->getId()], std::make_shared<IR::Immediate>(offset));
		auto deref = std::make_shared<IR::Memory>(expr, (IR::ir_segment)segment_register.getId(), (IR::ir_size)mem.getSize());

		// TempVar = ARG (possibly pop)
		auto temp_variable = IR::Variable::create_variable(mem.getSize());
		auto assign = std::make_shared<IR::Assign>(temp_variable, deref);
		context->instructions.push_back(assign);
		context->expression_map[symvar_arg->getId()] = temp_variable;
		symvar_arg->setAlias(temp_variable->get_name());
	}
	else if (this->is_fetch_arguments(lea_ast, context))
	{
		// lea_ast == VM_REG_X
		triton::arch::Register segment_register = mem.getConstSegmentRegister();
		if (segment_register.getId() == triton::arch::ID_REG_INVALID)
		{
			// DS?
			//segment_register = triton_api->registers.x86_ds;
		}
		triton::engines::symbolic::SharedSymbolicVariable symvar_source = get_symbolic_var(lea_ast);

		const triton::engines::symbolic::SharedSymbolicVariable symvar = triton_api->symbolizeMemory(mem);
		std::cout << "Deref(" << lea_ast << "," << segment_register.getName() << ")\n";

		// IR
		auto it = context->expression_map.find(symvar_source->getId());
		if (it == context->expression_map.end())
			throw std::runtime_error("what do you mean");

		// declare Temp
		auto temp_variable = IR::Variable::create_variable(mem.getSize());

		// Temp = memory(expr, segment, size)
		std::shared_ptr<IR::Expression> expr = it->second;
		std::shared_ptr<IR::Expression> deref = std::make_shared<IR::Memory>(expr, (IR::ir_segment)segment_register.getId(), (IR::ir_size)mem.getSize());
		context->instructions.push_back(std::make_shared<IR::Assign>(temp_variable, deref));
		context->expression_map[symvar->getId()] = temp_variable;
		symvar->setAlias(temp_variable->get_name());
	}
	else
	{
		std::cout << "unknown read addr: " << std::hex << mem_address << " " << lea_ast << std::endl;
	}
}
std::vector<std::shared_ptr<IR::Expression>> VMProtectAnalyzer::save_expressions(triton::arch::Instruction &triton_instruction, VMPHandlerContext *context)
{
	std::vector<std::shared_ptr<IR::Expression>> expressions;
	if (!is_unary_operation(triton_instruction) && !is_binary_operation(triton_instruction))
	{
		return expressions;
	}

	bool do_it = false;
	auto operand_index = 0;
	if (triton_instruction.getType() == triton::arch::x86::ID_INS_MUL
		|| triton_instruction.getType() == triton::arch::x86::ID_INS_IMUL)
	{
		if (triton_instruction.operands.size() == 1)
		{
			// edx:eax = eax * r/m
			triton::arch::Register _reg = triton_api->registers.x86_eax;
			switch (triton_instruction.operands[0].getSize())
			{
				case 1: _reg = triton_api->registers.x86_al; break;
				case 2: _reg = triton_api->registers.x86_ax; break;
				case 4: _reg = triton_api->registers.x86_eax; break;
				default: throw std::runtime_error("idk whats wrong");
			}
			const auto simplified_reg = triton_api->processSimplification(triton_api->getRegisterAst(_reg), true);
			if (simplified_reg->isSymbolized())
			{
				triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(simplified_reg);
				if (!_symvar)
					throw std::runtime_error("idk whats wrong2");

				// load symbolic
				auto _it = context->expression_map.find(_symvar->getId());
				if (_it != context->expression_map.end())
				{
					expressions.push_back(_it->second);
					do_it = true;
				}
			}
			else
			{
				expressions.push_back(std::make_shared<IR::Immediate>(
					triton_api->getConcreteRegisterValue(_reg).convert_to<triton::uint64>()));
			}
		}
		else if (triton_instruction.operands.size() == 3)
		{
			// op0 = r/m * imm
			operand_index = 1;
		}
	}

	for (; operand_index < triton_instruction.operands.size(); operand_index++)
	{
		const auto& operand = triton_instruction.operands[operand_index];
		if (operand.getType() == triton::arch::operand_e::OP_IMM)
		{
			expressions.push_back(std::make_shared<IR::Immediate>(
				operand.getConstImmediate().getValue()));
		}
		else if (operand.getType() == triton::arch::operand_e::OP_MEM)
		{
			const triton::arch::MemoryAccess& _mem = operand.getConstMemory();
			triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(triton_api->getMemoryAst(_mem), true));
			if (_symvar)
			{
				// load symbolic
				auto _it = context->expression_map.find(_symvar->getId());
				if (_it != context->expression_map.end())
				{
					expressions.push_back(_it->second);
					do_it = true;
					continue;
				}
			}

			// otherwise immediate
			expressions.push_back(std::make_shared<IR::Immediate>(
				triton_api->getConcreteMemoryValue(_mem).convert_to<triton::uint64>()));
		}
		else if (operand.getType() == triton::arch::operand_e::OP_REG)
		{
			const triton::arch::Register& _reg = operand.getConstRegister();
			triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(triton_api->getRegisterAst(_reg), true));
			if (_symvar)
			{
				if (_symvar->getId() == context->symvar_vmp_sp->getId())
				{
					// nope...
					do_it = false;
					break;
				}

				// load symbolic
				auto _it = context->expression_map.find(_symvar->getId());
				if (_it != context->expression_map.end())
				{
					expressions.push_back(_it->second);
					do_it = true;
					continue;
				}
			}

			// otherwise immediate
			expressions.push_back(std::make_shared<IR::Immediate>(
				triton_api->getConcreteRegisterValue(_reg).convert_to<triton::uint64>()));
		}
		else
			throw std::runtime_error("invalid operand type");
	}
	if (!do_it)
		expressions.clear();
	return expressions;
}
void VMProtectAnalyzer::check_arity_operation(triton::arch::Instruction& triton_instruction,
	const std::vector<std::shared_ptr<IR::Expression>>& operands_expressions, VMPHandlerContext* context, bool maybe_flag_written)
{
	if (triton_instruction.getType() == triton::arch::x86::ID_INS_CPUID)
	{
		std::shared_ptr<IR::Cpuid> statement = std::make_shared<IR::Cpuid>();
		context->instructions.push_back(statement);

		auto symvar_eax = this->triton_api->symbolizeRegister(triton_api->registers.x86_eax);
		auto symvar_ebx = this->triton_api->symbolizeRegister(triton_api->registers.x86_ebx);
		auto symvar_ecx = this->triton_api->symbolizeRegister(triton_api->registers.x86_ecx);
		auto symvar_edx = this->triton_api->symbolizeRegister(triton_api->registers.x86_edx);
		context->expression_map[symvar_eax->getId()] = std::make_shared<IR::Register>(triton_api->registers.x86_eax);
		context->expression_map[symvar_ebx->getId()] = std::make_shared<IR::Register>(triton_api->registers.x86_ebx);
		context->expression_map[symvar_ecx->getId()] = std::make_shared<IR::Register>(triton_api->registers.x86_ecx);
		context->expression_map[symvar_edx->getId()] = std::make_shared<IR::Register>(triton_api->registers.x86_edx);
		symvar_eax->setAlias("cpuid_eax");
		symvar_ebx->setAlias("cpuid_ebx");
		symvar_ecx->setAlias("cpuid_ecx");
		symvar_edx->setAlias("cpuid_edx");
		return;
	}
	else if (triton_instruction.getType() == triton::arch::x86::ID_INS_RDTSC)
	{
		std::shared_ptr<IR::Instruction> statement = std::make_shared<IR::Rdtsc>();
		context->instructions.push_back(statement);

		auto symvar_eax = this->triton_api->symbolizeRegister(triton_api->registers.x86_eax);
		auto symvar_edx = this->triton_api->symbolizeRegister(triton_api->registers.x86_edx);
		context->expression_map[symvar_eax->getId()] = std::make_shared<IR::Register>(triton_api->registers.x86_eax);
		context->expression_map[symvar_edx->getId()] = std::make_shared<IR::Register>(triton_api->registers.x86_edx);
		symvar_eax->setAlias("rdtsc_eax");
		symvar_edx->setAlias("rdtsc_edx");
		return;
	}

	bool unary = is_unary_operation(triton_instruction) && operands_expressions.size() == 1;
	bool binary = is_binary_operation(triton_instruction) && operands_expressions.size() == 2;
	if (!unary && !binary)
		return;

	//
	if ((triton_instruction.getType() == triton::arch::x86::ID_INS_MUL
		|| triton_instruction.getType() == triton::arch::x86::ID_INS_IMUL) && triton_instruction.operands.size() == 1)
	{
		// edx:eax = eax * r/m
		triton::arch::Register _reg_eax, _reg_edx;
		switch (triton_instruction.operands[0].getSize())
		{
			case 1:
			{
				_reg_eax = triton_api->registers.x86_ax;
				break;
			}
			case 2:
			{
				_reg_eax = triton_api->registers.x86_ax;
				_reg_edx = triton_api->registers.x86_dx;
				break;
			}
			case 4:
			{
				_reg_eax = triton_api->registers.x86_eax;
				_reg_edx = triton_api->registers.x86_edx;
				break;
			}
			default: throw std::runtime_error("idk whats wrong");
		}

		if (2 <= triton_instruction.operands[0].getSize())
		{
			// t0 = mul(eax, src)
			// t1 = extract(t0)
			// t2 = extract(t0)			but... is this good idea?
		}
		return;
	}

	// symbolize destination
	triton::engines::symbolic::SharedSymbolicVariable symvar;
	const auto& operand0 = triton_instruction.operands[0];
	if (operand0.getType() == triton::arch::operand_e::OP_REG)
	{
		const triton::arch::Register& _reg = operand0.getConstRegister();
		triton_api->concretizeRegister(_reg);
		symvar = triton_api->symbolizeRegister(_reg);
	}
	else if (operand0.getType() == triton::arch::operand_e::OP_MEM)
	{
		const triton::arch::MemoryAccess& _mem = operand0.getConstMemory();
		triton_api->concretizeMemory(_mem);
		symvar = triton_api->symbolizeMemory(_mem);
	}
	else
	{
		throw std::runtime_error("invalid operand type");
	}


	std::shared_ptr<IR::Variable> temp_variable = IR::Variable::create_variable(operand0.getSize());
	std::shared_ptr<IR::Expression> expr;
	if (unary)
	{
		// unary
		auto op0_expression = operands_expressions[0];
		switch (triton_instruction.getType())
		{
			case triton::arch::x86::ID_INS_INC:
			{
				expr = std::make_shared<IR::Inc>(op0_expression);
				break;
			}
			case triton::arch::x86::ID_INS_DEC:
			{
				expr = std::make_shared<IR::Dec>(op0_expression);
				break;
			}
			case triton::arch::x86::ID_INS_NEG:
			{
				expr = std::make_shared<IR::Neg>(op0_expression);
				break;
			}
			case triton::arch::x86::ID_INS_NOT:
			{
				expr = std::make_shared<IR::Not>(op0_expression);
				break;
			}
			default:
			{
				throw std::runtime_error("unknown unary operation");
			}
		}
	}
	else
	{
		// binary
		auto op0_expression = operands_expressions[0];
		auto op1_expression = operands_expressions[1];
		switch (triton_instruction.getType())
		{
			case triton::arch::x86::ID_INS_ADD:
			{
				expr = std::make_shared<IR::Add>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_SUB:
			{
				expr = std::make_shared<IR::Sub>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_SHL:
			{
				expr = std::make_shared<IR::Shl>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_SHR:
			{
				expr = std::make_shared<IR::Shr>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_RCR:
			{
				expr = std::make_shared<IR::Rcr>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_RCL:
			{
				expr = std::make_shared<IR::Rcl>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_ROL:
			{
				expr = std::make_shared<IR::Rol>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_ROR:
			{
				expr = std::make_shared<IR::Ror>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_AND:
			{
				expr = std::make_shared<IR::And>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_OR:
			{
				expr = std::make_shared<IR::Or>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_XOR:
			{
				expr = std::make_shared<IR::Xor>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_CMP:
			{
				expr = std::make_shared<IR::Cmp>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_TEST:
			{
				expr = std::make_shared<IR::Test>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_MUL:
			{
				expr = std::make_shared<IR::Mul>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_IMUL:
			{
				expr = std::make_shared<IR::IMul>(op0_expression, op1_expression);
				break;
			}
			default:
			{
				throw std::runtime_error("unknown binary operation");
			}
		}
	}
	context->instructions.push_back(std::make_shared<IR::Assign>(temp_variable, expr));
	context->expression_map[symvar->getId()] = temp_variable;
	symvar->setAlias(temp_variable->get_name());

	// check if flags are written
	if (maybe_flag_written)
	{
		// declare Temp
		auto _temp = IR::Variable::create_variable(4);

		// tvar = FlagOf(expr)
		auto symvar_eflags = this->triton_api->symbolizeRegister(this->triton_api->registers.x86_eflags);
		context->instructions.push_back(std::make_shared<IR::Assign>(_temp, std::make_shared<IR::FlagsOf>(expr)));
		context->expression_map[symvar_eflags->getId()] = _temp;
		symvar_eflags->setAlias(_temp->get_name());
	}
}
void VMProtectAnalyzer::check_store_access(triton::arch::Instruction &triton_instruction, VMPHandlerContext *context)
{
	const auto& storeAccess = triton_instruction.getStoreAccess();
	for (const std::pair<triton::arch::MemoryAccess, triton::ast::SharedAbstractNode>& pair : storeAccess)
	{
		const triton::arch::MemoryAccess &mem = pair.first;
		//const triton::ast::SharedAbstractNode &mem_ast = pair.second;
		const triton::ast::SharedAbstractNode &mem_ast = triton_api->getMemoryAst(mem);
		const triton::uint64 address = mem.getAddress();
		triton::ast::SharedAbstractNode lea_ast = mem.getLeaAst();
		if (!lea_ast)
		{
			// most likely can be ignored
			continue;
		}

		lea_ast = triton_api->processSimplification(lea_ast, true);
		if (!lea_ast->isSymbolized())
		{
			// most likely can be ignored
			continue;
		}

		if (this->is_scratch_area_address(lea_ast, context))
		{
			const triton::uint64 scratch_offset = lea_ast->evaluate().convert_to<triton::uint64>() - context->x86_sp;
			std::cout << "Store [x86_sp + 0x" << std::hex << scratch_offset << "]\n";

			// create IR (VM_REG = mem_ast)
			auto source_node = triton_api->processSimplification(mem_ast, true);
			triton::engines::symbolic::SharedSymbolicVariable symvar = get_symbolic_var(source_node);
			if (symvar)
			{
				auto ir_imm = std::make_shared<IR::Immediate>(scratch_offset);
				std::shared_ptr<IR::Expression> v1 = std::make_shared<IR::Memory>(ir_imm, IR::ir_segment_scratch, (IR::ir_size)mem.getSize());
				auto it = context->expression_map.find(symvar->getId());
				if (it != context->expression_map.end())
				{
					std::shared_ptr<IR::Expression> expr = it->second;
					context->instructions.push_back(std::make_shared<IR::Assign>(v1, expr));
				}
				else if (symvar->getId() == context->symvar_vmp_sp->getId())
				{
					std::shared_ptr<IR::Expression> expr = std::make_shared<IR::Register>(this->get_sp_register());
					context->instructions.push_back(std::make_shared<IR::Assign>(v1, expr));
				}
				else if (symvar->getAlias().find("eflags") != std::string::npos)
				{
					std::shared_ptr<IR::Expression> expr = std::make_shared<IR::Register>(triton_api->registers.x86_eflags);
					context->instructions.push_back(std::make_shared<IR::Assign>(v1, expr));
				}
				else
				{
					printf("%s\n", symvar->getAlias().c_str());
					throw std::runtime_error("what do you mean 2");
				}
			}
			else
			{
				std::cout << "source_node: " << source_node << std::endl;
			}
		}
		else if (this->is_stack_address(lea_ast, context))
		{
			// stores to stack
			const triton::uint64 stack_offset = address - context->vmp_sp;

			std::shared_ptr<IR::Expression> expr;
			auto get_expr = [this, context](std::shared_ptr<triton::API> ctx, triton::ast::SharedAbstractNode mem_ast)
			{
				std::shared_ptr<IR::Expression> expr;
				auto simplified_source_node = ctx->processSimplification(mem_ast, true);
				if (!simplified_source_node->isSymbolized())
				{
					// expression is immediate
					expr = std::make_shared<IR::Immediate>(simplified_source_node->evaluate().convert_to<triton::uint64>());
				}
				else
				{
					triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(simplified_source_node);
					if (_symvar)
					{
						auto _it = context->expression_map.find(_symvar->getId());
						if (_it == context->expression_map.end())
						{
							throw std::runtime_error("what do you mean...");
						}
						expr = _it->second;
					}
				}
				return expr;
			};
			expr = get_expr(this->triton_api, mem_ast);
			if (!expr && mem.getSize() == 2)
			{
				const triton::arch::MemoryAccess _mem(mem.getAddress(), 1);
				expr = get_expr(this->triton_api, triton_api->getMemoryAst(_mem));
			}

			// should be push
			if (expr)
			{
				auto ir_stack = context->expression_map[context->symvar_vmp_sp->getId()];
				auto ir_stack_address = std::make_shared<IR::Add>(ir_stack, std::make_shared<IR::Immediate>(stack_offset));

				std::shared_ptr<IR::Expression> v1 = std::make_shared<IR::Memory>(
					ir_stack_address, (IR::ir_segment)mem.getConstSegmentRegister().getId(), (IR::ir_size)mem.getSize());
				context->instructions.push_back(std::make_shared<IR::Assign>(v1, expr));
			}
			else
			{
				std::cout << "unknown store addr: " << std::hex << address << ", lea_ast: " << lea_ast 
					<< ", simplified_source_node: " << triton_api->processSimplification(mem_ast, true) << std::endl;
			}
		}
		else
		{
			// create IR (VM_REG = mem_ast)
			// get right expression
			std::shared_ptr<IR::Expression> expr;
			auto simplified_source_node = triton_api->processSimplification(mem_ast, true);
			if (!simplified_source_node->isSymbolized())
			{
				// expression is immediate
				expr = std::make_shared<IR::Immediate>(simplified_source_node->evaluate().convert_to<triton::uint64>());
			}
			else
			{
				triton::engines::symbolic::SharedSymbolicVariable symvar1 = get_symbolic_var(simplified_source_node);
				if (symvar1)
				{
					auto _it = context->expression_map.find(symvar1->getId());
					if (_it == context->expression_map.end())
					{
						throw std::runtime_error("what do you mean...");
					}
					expr = _it->second;
				}
			}

			triton::engines::symbolic::SharedSymbolicVariable symvar0 = get_symbolic_var(lea_ast);
			if (symvar0 && expr)
			{
				auto it0 = context->expression_map.find(symvar0->getId());
				if (it0 != context->expression_map.end())
				{
					std::shared_ptr<IR::Expression> v1 = std::make_shared<IR::Memory>(it0->second, 
						(IR::ir_segment)mem.getConstSegmentRegister().getId(), (IR::ir_size)mem.getSize());
					context->instructions.push_back(std::make_shared<IR::Assign>(v1, expr));
				}
				else
				{
					throw std::runtime_error("what do you mean 2");
				}
			}
			else
			{
				std::cout << "unknown store addr: " << std::hex << address << ", lea_ast: " << lea_ast << ", simplified_source_node: " << simplified_source_node << std::endl;
			}
		}
	}
}

void VMProtectAnalyzer::analyze_vm_handler(AbstractStream& stream, triton::uint64 handler_address)
{
	//this->m_scratch_size = 0xC0; // test

	// reset triton api
	triton_api->clearCallbacks();
	triton_api->concretizeAllMemory();
	triton_api->concretizeAllRegister();

	// allocate scratch area
	const triton::arch::Register bp_register = this->get_bp_register();
	const triton::arch::Register sp_register = this->get_sp_register();
	const triton::arch::Register si_register = this->is_x64() ? triton_api->registers.x86_rsi : triton_api->registers.x86_esi;
	const triton::arch::Register ip_register = this->get_ip_register();

	constexpr unsigned long c_stack_base = 0x1000;
	triton_api->setConcreteRegisterValue(bp_register, c_stack_base);
	triton_api->setConcreteRegisterValue(sp_register, c_stack_base - this->m_scratch_size);

	unsigned int arg0 = c_stack_base;
	triton_api->setConcreteMemoryAreaValue(c_stack_base, (const triton::uint8*)&arg0, 4);

	// ebp = VM's "stack" pointer
	triton::engines::symbolic::SharedSymbolicVariable symvar_vmp_sp = triton_api->symbolizeRegister(bp_register);

	// esi = pointer to VM bytecode
	triton::engines::symbolic::SharedSymbolicVariable symvar_bytecode = triton_api->symbolizeRegister(si_register);

	// x86 stack pointer
	triton::engines::symbolic::SharedSymbolicVariable symvar_x86_sp = triton_api->symbolizeRegister(sp_register);

	symvar_vmp_sp->setAlias("vmp_sp");
	symvar_bytecode->setAlias("bytecode");
	symvar_x86_sp->setAlias("x86_sp");

	// yo...
	VMPHandlerContext context;
	context.scratch_area_size = this->is_x64() ? 0x140 : 0x60;
	context.address = handler_address;
	context.vmp_sp = triton_api->getConcreteRegisterValue(bp_register).convert_to<triton::uint64>();
	context.bytecode = triton_api->getConcreteRegisterValue(si_register).convert_to<triton::uint64>();
	context.x86_sp = triton_api->getConcreteRegisterValue(sp_register).convert_to<triton::uint64>();
	context.symvar_vmp_sp = symvar_vmp_sp;
	context.symvar_bytecode = symvar_bytecode;
	context.symvar_x86_sp = symvar_x86_sp;

	// expr
	std::shared_ptr<IR::Expression> vmp_sp = std::make_shared<IR::Variable>("vmp_sp", (IR::ir_size)sp_register.getSize());
	std::shared_ptr<IR::Expression> x86_sp = std::make_shared<IR::Variable>("x86_sp", (IR::ir_size)sp_register.getSize());
	context.expression_map.insert(std::make_pair(symvar_vmp_sp->getId(), vmp_sp));
	context.expression_map.insert(std::make_pair(symvar_x86_sp->getId(), x86_sp));

	// cache basic block (maybe not best place)
	std::shared_ptr<BasicBlock> basic_block;
	auto handler_it = this->m_handlers.find(handler_address);
	if (handler_it == this->m_handlers.end())
	{
		basic_block = make_cfg(stream, handler_address);
		this->m_handlers.insert(std::make_pair(handler_address, basic_block));
	}
	else
	{
		basic_block = handler_it->second;
	}

	triton::uint64 expected_return_address = 0;
	for (auto it = basic_block->instructions.begin(); it != basic_block->instructions.end();)
	{
		const std::shared_ptr<x86_instruction> xed_instruction = *it;
		const std::vector<xed_uint8_t> bytes = xed_instruction->get_bytes();
		bool mem_read = false;
		for (xed_uint_t j = 0, memops = xed_instruction->get_number_of_memory_operands(); j < memops; j++)
		{
			if (xed_instruction->is_mem_read(j))
			{
				mem_read = true;
				break;
			}
		}

		// triton removes from written registers if it is NOT actually written, so xed helps here
		const bool maybe_flag_written = xed_instruction->writes_flags();

		// do stuff with triton
		triton::arch::Instruction triton_instruction;
		triton_instruction.setOpcode(&bytes[0], (triton::uint32)bytes.size());
		triton_instruction.setAddress(xed_instruction->get_addr());

		// fix ip
		triton_api->setConcreteRegisterValue(ip_register, xed_instruction->get_addr());

		// DIS
		triton_api->disassembly(triton_instruction);
		if (mem_read 
			&& (triton_instruction.getType() != triton::arch::x86::ID_INS_POP
				&& triton_instruction.getType() != triton::arch::x86::ID_INS_POPFD)) // no need but makes life easier
		{
			for (auto& operand : triton_instruction.operands)
			{
				if (operand.getType() == triton::arch::OP_MEM)
				{
					triton_api->getSymbolicEngine()->initLeaAst(operand.getMemory());
					this->symbolize_memory(operand.getConstMemory(), &context);
				}
			}
		}
		std::vector<std::shared_ptr<IR::Expression>> operands_expressions = this->save_expressions(triton_instruction, &context);
		if (!triton_api->processing(triton_instruction))
		{
			throw std::runtime_error("triton processing failed");
		}

		// lol
		this->check_arity_operation(triton_instruction, operands_expressions, &context, maybe_flag_written);
		this->check_store_access(triton_instruction, &context);

		if (xed_instruction->get_category() != XED_CATEGORY_UNCOND_BR
			|| xed_instruction->get_branch_displacement_width() == 0)
		{
			std::cout << "\t" << triton_instruction << "\n";
		}

		// symbolize eflags
		static std::string ins_name;
		for (const auto& pair : xed_instruction->get_written_registers())
		{
			if (pair.is_flag())
			{
				ins_name = xed_instruction->get_name();
				break;
			}
		}

		// pushfd/pushfq
		if (triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFD
			|| triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFQ)
		{
			auto eflags_ast = this->triton_api->getRegisterAst(this->triton_api->registers.x86_eflags);
			triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(eflags_ast, true));
			if (_symvar)
			{
				auto it = context.expression_map.find(_symvar->getId());
				if (it == context.expression_map.end())
				{
					// ?
					throw std::runtime_error("bluh");
				}

				triton::arch::MemoryAccess _mem(this->get_sp(), triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFD ? 4 : 8);
				auto _symvar = triton_api->symbolizeMemory(_mem);
				context.expression_map[_symvar->getId()] = it->second;
			}
		}

		if (++it != basic_block->instructions.end())
		{
			// loop until it reaches end
			continue;
		}

		if (triton_instruction.getType() == triton::arch::x86::ID_INS_CALL)
		{
			expected_return_address = xed_instruction->get_addr() + 5;
		}
		else if (triton_instruction.getType() == triton::arch::x86::ID_INS_RET)
		{
			if (expected_return_address != 0 && this->get_ip() == expected_return_address)
			{
				basic_block = make_cfg(stream, expected_return_address);
				it = basic_block->instructions.begin();
			}
		}

		while (it == basic_block->instructions.end())
		{
			if (basic_block->next_basic_block && basic_block->target_basic_block)
			{
				// it ends with conditional branch
				if (triton_instruction.isConditionTaken())
				{
					basic_block = basic_block->target_basic_block;
				}
				else
				{
					basic_block = basic_block->next_basic_block;
				}
			}
			else if (basic_block->target_basic_block)
			{
				// it ends with jmp?
				basic_block = basic_block->target_basic_block;
			}
			else if (basic_block->next_basic_block)
			{
				// just follow :)
				basic_block = basic_block->next_basic_block;
			}
			else
			{
				// perhaps finishes?
				goto l_categorize_handler;
			}
			it = basic_block->instructions.begin();
		}
	}

l_categorize_handler:
	this->categorize_handler(&context);
	this->m_vmp_instructions.insert(this->m_vmp_instructions.end(), context.instructions.begin(), context.instructions.end());
}
void VMProtectAnalyzer::analyze_vm_exit(VMPHandlerContext* context)
{
	// not the best impl but works
	std::stack<triton::arch::Register> modified_registers;
	const triton::arch::Register bp_register = this->get_bp_register();
	const triton::uint64 previous_stack = triton_api->getConcreteRegisterValue(bp_register).convert_to<triton::uint64>();

	std::shared_ptr<BasicBlock> basic_block = this->m_handlers[context->address];
	for (auto it = basic_block->instructions.begin(); it != basic_block->instructions.end();)
	{
		const auto xed_instruction = *it;
		const std::vector<xed_uint8_t> bytes = xed_instruction->get_bytes();

		// do stuff with triton
		triton::arch::Instruction triton_instruction;
		triton_instruction.setOpcode(&bytes[0], (triton::uint32)bytes.size());
		triton_instruction.setAddress(xed_instruction->get_addr());
		if (!triton_api->processing(triton_instruction))
		{
			throw std::runtime_error("triton processing failed");
		}

		for (const auto& pair : triton_instruction.getWrittenRegisters())
		{
			const triton::arch::Register& _reg = pair.first;
			if (this->is_x64())
			{
				if (_reg.getParent() == triton::arch::ID_REG_X86_RSP
					|| _reg.getParent() == triton::arch::ID_REG_X86_RIP)
				{
					continue;
				}
			}
			else
			{
				if (_reg.getParent() == triton::arch::ID_REG_X86_ESP
					|| _reg.getParent() == triton::arch::ID_REG_X86_EIP)
				{
					continue;
				}
			}

			// flags -> eflags
			if (this->triton_api->isFlag(_reg))
			{
				modified_registers.push(this->triton_api->registers.x86_eflags);
			}
			else if (this->is_x64())
			{
				if (_reg.getSize() == 8)
				{
					modified_registers.push(_reg);
				}
			}
			else if (_reg.getSize() == 4)
			{
				modified_registers.push(_reg);
			}
		}

		if (++it != basic_block->instructions.end())
		{
			// loop until it reaches end
			//std::cout << triton_instruction << "\n";
			continue;
		}

		if (basic_block->next_basic_block && basic_block->target_basic_block)
		{
			// it ends with conditional branch
			if (triton_instruction.isConditionTaken())
			{
				basic_block = basic_block->target_basic_block;
			}
			else
			{
				basic_block = basic_block->next_basic_block;
			}
		}
		else if (basic_block->target_basic_block)
		{
			// it ends with jmp?
			basic_block = basic_block->target_basic_block;
		}
		else if (basic_block->next_basic_block)
		{
			// just follow :)
			basic_block = basic_block->next_basic_block;
		}
		else
		{
			// perhaps finishes?
			break;
		}

		it = basic_block->instructions.begin();
	}

	std::set<triton::arch::Register> _set;
	std::stack<triton::arch::Register> _final;
	while (!modified_registers.empty())
	{
		triton::arch::Register r = modified_registers.top();
		modified_registers.pop();

		if (_set.count(r) == 0)
		{
			_set.insert(r);
			_final.push(r);
		}
	}

	context->instructions.clear();
	while (!_final.empty())
	{
		triton::arch::Register r = _final.top();
		_final.pop();

		auto _pop = std::make_shared<IR::Pop>(std::make_shared<IR::Register>(r));
		context->instructions.push_back(std::move(_pop));
	}
	context->instructions.push_back(std::make_shared<IR::Ret>());
}
void VMProtectAnalyzer::categorize_handler(VMPHandlerContext *context)
{
	const triton::arch::Register bp_register = this->is_x64() ? triton_api->registers.x86_rbp : triton_api->registers.x86_ebp;
	const triton::arch::Register sp_register = this->is_x64() ? triton_api->registers.x86_rsp : triton_api->registers.x86_esp;
	const triton::arch::Register si_register = this->is_x64() ? triton_api->registers.x86_rsi : triton_api->registers.x86_esi;
	const triton::uint64 bytecode = triton_api->getConcreteRegisterValue(si_register).convert_to<triton::uint64>();
	const triton::uint64 x86_sp = this->get_sp();
	const triton::uint64 vmp_sp = this->get_bp();

	std::cout << std::hex << "handlers outputs:\n"
		<< "\tbytecode: 0x" << context->bytecode << " -> 0x" << bytecode << "\n"
		<< "\t  x86_sp: 0x" << context->x86_sp << " -> 0x" << x86_sp << "\n"
		<< "\t  vmp_sp: 0x" << context->vmp_sp << " -> 0x" << vmp_sp << "\n";

	bool handler_detected = false;

	// check x86_sp
	const triton::ast::SharedAbstractNode simplified_x86_sp =
		triton_api->processSimplification(triton_api->getRegisterAst(sp_register), true);
	std::set<triton::ast::SharedAbstractNode> symvars = collect_symvars(simplified_x86_sp);
	if (symvars.size() == 1)
	{
		const triton::ast::SharedAbstractNode _node = *symvars.begin();
		const auto _symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(_node)->getSymbolicVariable();
		if (_symvar->getId() == context->symvar_vmp_sp->getId())
		{
			// if x86_sp == compute(vmp_sp) then vm exit handler
			this->analyze_vm_exit(context);
			handler_detected = true;
			return;
		}
	}

	// check vmp_sp
	const triton::ast::SharedAbstractNode simplified_vmp_sp =
		triton_api->processSimplification(triton_api->getRegisterAst(bp_register), true);
	if (simplified_vmp_sp->getType() == triton::ast::BVADD_NODE)
	{
		// vmp_sp = add(vmp_sp, vmp_sp_offset)
		triton::sint64 vmp_sp_offset = this->get_bp() - context->vmp_sp;	// needs to be signed
		std::shared_ptr<IR::Expression> vmp_sp_expr = context->expression_map[context->symvar_vmp_sp->getId()];
		std::shared_ptr<IR::Expression> add_expr = std::make_shared<IR::Add>(vmp_sp_expr, std::make_shared<IR::Immediate>(vmp_sp_offset));
		context->instructions.push_back(std::make_shared<IR::Assign>(vmp_sp_expr, add_expr));
	}
	else if (simplified_vmp_sp->getType() == triton::ast::VARIABLE_NODE)
	{
		const auto _symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(simplified_vmp_sp)->getSymbolicVariable();
		if (_symvar != context->symvar_vmp_sp)
		{
			auto it = context->expression_map.find(_symvar->getId());
			if (it == context->expression_map.end())
			{
				throw std::runtime_error("invalid vmp_sp");
			}

			// vmp_sp = x
			std::shared_ptr<IR::Expression> vmp_sp_expr = context->expression_map[context->symvar_vmp_sp->getId()];
			context->instructions.push_back(std::make_shared<IR::Assign>(vmp_sp_expr, it->second));
		}
	}
	else
	{
		std::cout << simplified_vmp_sp << std::endl;
		throw std::runtime_error("invalid vmp_sp");
	}

	if (!handler_detected)
	{
		//this->print_output();
		//output_strings.clear();
		//getchar();
	}
}




void VMProtectAnalyzer::print_output()
{
	// so... apply simplification?

	// "vmp_sp" has different pointer now so something need to be done

	for (const std::shared_ptr<IR::Instruction>& ins : m_vmp_instructions)
	{
		std::cout << "\t" << ins << "\n";
	}
	std::cout << std::endl;
}