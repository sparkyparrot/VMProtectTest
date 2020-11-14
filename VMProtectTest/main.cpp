﻿#include "pch.h"

#pragma comment(lib, "xed.lib")
#pragma comment(lib, "triton.lib")

#include <TlHelp32.h>
#include <tchar.h>
#include "VMProtectAnalyzer.hpp"
#include "ProcessStream.hpp"

DWORD find_process(LPCTSTR processName)
{
	DWORD processId = 0;
	HANDLE hProcessSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (hProcessSnap != INVALID_HANDLE_VALUE)
	{
		// Set the size of the structure before using it.
		PROCESSENTRY32 pe32;
		pe32.dwSize = sizeof(PROCESSENTRY32);
		if (Process32First(hProcessSnap, &pe32))
		{
			do
			{
				static size_t c_nLength = _tcslen(processName);
				size_t nLength = _tcslen(pe32.szExeFile);
				if (nLength < c_nLength ||
					_tcscmp(pe32.szExeFile + nLength - c_nLength, processName) != 0)
				{
					continue;
				}

				processId = pe32.th32ProcessID;
				break;
			} while (Process32Next(hProcessSnap, &pe32));
		}
		CloseHandle(hProcessSnap);
	}
	return processId;
}



void vmp_ultimate()
{
	ProcessStream stream;
	if (!stream.open(0x5d98))
		throw std::runtime_error("stream.open failed.");


	VMProtectAnalyzer analyzer;
	analyzer.load(stream, 0x00400000, 0x34000, 0x21CEE5);	// vmp0


	analyzer.analyze_vm_enter(stream, 0x00401520);		// IsValidImageCRC

	//analyzer.analyze_vm_enter(stream, 0x00401450);
	//analyzer.analyze_vm_enter(stream, 0x004014F0); // ultra
	//analyzer.analyze_vm_enter(stream, 0x00401490); // mutation

	unsigned long long handler_address = analyzer.get_ip();
	std::cout << std::hex << handler_address << std::endl;
	while (0x00400000 <= handler_address && handler_address <= (0x00400000 + 0x34000 + 0x21CEE5))
	{
		analyzer.analyze_vm_handler(stream, handler_address);
		std::cout << std::endl << std::endl << std::endl << std::endl;
		handler_address = analyzer.get_ip();
	}

	// idk
	std::cout << std::endl << std::endl;
	analyzer.print_output();
}

void test_v1()
{
	DWORD processId = find_process(L"devirtualizeme32_vmp_3.0.9_v1.exe");
	printf("pid: %08X\n", processId);

	ProcessStream stream;
	if (!stream.open(processId))
		throw std::runtime_error("stream.open failed.");

	VMProtectAnalyzer analyzer;
	analyzer.load(stream, 0x00400000, 0x17000, 0x86CB0);
	//analyzer.analyze_vm_enter(stream, 0x0040C890);
	analyzer.analyze_vm_enter(stream, 0x004312D7);
	//analyzer.analyze_vm_enter(stream, 0x0041F618);				//ok
	//analyzer.analyze_vm_enter(stream, 0x00477CBB);				//ok

	triton::uint64 handler_address = analyzer.get_ip();
	while (handler_address)
	{
		std::cout << "start analyzing " << std::hex << handler_address << "\n";
		analyzer.analyze_vm_handler(stream, handler_address);
		std::cout << "done.\n\n\n\n";
		handler_address = analyzer.get_ip();
	}
	analyzer.print_output();
}
void test_x86_64()
{
	DWORD processId = find_process(L"devirtualizeme64_vmp_3.0.9_v1.exe");
	printf("pid: %08X\n", processId);

	ProcessStream stream(true);
	if (!stream.open(processId))
		throw std::runtime_error("stream.open failed.");

	unsigned long long module_base = 0x140000000ull;

	VMProtectAnalyzer analyzer(triton::arch::ARCH_X86_64);
	analyzer.load(stream, module_base, 0x1C000, 0xE1F74);		// vmp0
	analyzer.load(stream, module_base, 0x1B000, 0xA80);			// pdata

	//analyzer.analyze_vm_enter(stream, 0x1400FD439);
	//analyzer.analyze_vm_enter(stream, 0x1400FD443);
	analyzer.analyze_vm_enter(stream, 0x1400FD44D);
	//analyzer.analyze_vm_enter(stream, 0x1400FD457ull); // after messagebox

	triton::uint64 handler_address = analyzer.get_ip();
	while (handler_address)
	{
		std::cout << "start analyzing " << std::hex << handler_address << "\n";
		analyzer.analyze_vm_handler(stream, handler_address);
		std::cout << "done.\n\n\n\n";
		handler_address = analyzer.get_ip();
	}
	analyzer.print_output();
}

void test_demo()
{
	DWORD processId = find_process(L"Demo.vmp.exe");
	printf("pid: %08X\n", processId);

	ProcessStream stream;
	if (!stream.open(processId))
		throw std::runtime_error("stream.open failed.");

	VMProtectAnalyzer analyzer;
	analyzer.load(stream, 0x01150000, 0x4000, 0x8D620);

	if (false)
	{
		// run through
		analyzer.analyze_vm_enter(stream, 0x01151020);
		triton::uint64 handler_address = analyzer.get_ip();
		std::cout << std::hex << handler_address << std::endl;
		while (handler_address)
		{
			analyzer.analyze_vm_handler(stream, handler_address);
			std::cout << std::endl << std::endl << std::endl << std::endl;
			handler_address = analyzer.get_ip();
		}
	}
	else
	{
		// test handlers
		for (int i = 20; i <= 0xFF; i++)
		{
			triton::uint64 handler_address = 0;
			stream.seek(0x011C4294 + (i * 4));
			stream.read(&handler_address, 4);

			std::cout << i << " " << std::hex << handler_address << std::endl;
			analyzer.analyze_vm_handler(stream, handler_address);
			std::cout << std::endl << std::endl << std::endl << std::endl;
		}
	}
	// idk
	std::cout << std::endl << std::endl;
	analyzer.print_output();
}



extern void runtime_optimize(AbstractStream& stream,
	triton::uint64 address, triton::uint64 module_base, triton::uint64 section_addr, triton::uint64 section_size);
void t()
{
	constexpr bool x86_64 = 1;
	DWORD processId;
	if (x86_64)
		processId = find_process(L"devirtualizeme64_vmp_3.0.9_v1.exe");
	else
		processId = find_process(L"devirtualizeme32_vmp_3.0.9_v1.exe");
	printf("pid: %08X\n", processId);

	ProcessStream stream(x86_64);
	if (!stream.open(processId))
		throw std::runtime_error("stream.open failed.");

	if (x86_64)
		runtime_optimize(stream, 0x1400FD439ull, 0x140000000ull, 0x1B000, 0xA80);
	else
		runtime_optimize(stream, 0x4312d7, 0x00400000, 0x17000, 0x86CB0);
}





int main()
{
	// Once, before using Intel XED, you must call xed_tables_init() to initialize the tables Intel XED uses for encoding and decoding:
	xed_tables_init();

	try
	{
		//test_demo();
		//test_v1();
		test_x86_64();
		//t();
	}
	catch (const std::exception &ex)
	{
		std::cout << ex.what() << std::endl;
	}
	return 0;
}