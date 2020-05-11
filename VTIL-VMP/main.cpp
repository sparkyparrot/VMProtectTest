#include <iostream>

#include <vtil/vtil>
#pragma comment(lib, "VTIL-Architecture.lib")
#pragma comment(lib, "VTIL-Common.lib")
#pragma comment(lib, "VTIL-SymEx.lib")

void create_basic_block()
{
	auto bb = vtil::basic_block::begin(0);

	// vmenter
	bb->push<uint32_t>(0x9486CFEA);
	bb->push<uint32_t>(0x4869C5);
	bb->push(vtil::register_desc(vtil::register_physical, X86_REG_ECX, 32));
	bb->push(vtil::register_desc(vtil::register_physical, X86_REG_EBP, 32));
	bb->push(vtil::register_desc(vtil::register_physical, X86_REG_EAX, 32));
	bb->push(vtil::register_desc(vtil::register_physical, X86_REG_EDI, 32));
	bb->push(vtil::register_desc(vtil::register_physical, X86_REG_EFLAGS, 32));
	bb->push(vtil::register_desc(vtil::register_physical, X86_REG_EBX, 32));
	bb->push(vtil::register_desc(vtil::register_physical, X86_REG_EDX, 32));
	bb->push(vtil::register_desc(vtil::register_physical, X86_REG_ESI, 32));
	bb->push<uint32_t>(0);

	// vm1
	bb->pop(vtil::register_desc(0, 0, 32));
	bb->pop(vtil::register_desc(0, 1, 32));
	bb->pop(vtil::register_desc(0, 2, 32));
	bb->pop(vtil::register_desc(0, 3, 32));
	bb->pop(vtil::register_desc(0, 4, 32));
	bb->pop(vtil::register_desc(0, 5, 32));
	bb->pop(vtil::register_desc(0, 6, 32));
	bb->pop(vtil::register_desc(0, 7, 32));
	bb->pop(vtil::register_desc(0, 8, 32));
	bb->pop(vtil::register_desc(0, 9, 32));
	bb->pop(vtil::register_desc(0, 10, 32));
	bb->pop(vtil::register_desc(0, 10, 32));

	//

	for (vtil::instruction &x : *bb)
	{
		std::cout << x.to_string() << std::endl;
	}
}

void create_symbolic_expression()
{
	/*vtil::symbolic::unique_identifier id("vmp_stack");
	vtil::symbolic::expression expr1(id, 32);
	std::cout << expr1.to_string() << std::endl;

	auto expr = vtil::symbolic::expression::make(
		expr1, vtil::math::operator_id::add, vtil::symbolic::expression(8, 32));

	auto expr2 = vtil::symbolic::expression::make(
		expr, vtil::math::operator_id::add, vtil::symbolic::expression(8, 32));


	std::cout << expr2.to_string() << std::endl;
	std::cout << expr2.simplify().to_string() << std::endl;*/
}

int main()
{
	create_basic_block();
	return 0;
}