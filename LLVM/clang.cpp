#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
__declspec (noinline)
#elif defined(__clang__)
__attribute__((noinline))
#endif
void push(unsigned int a)
{
    printf("push %d\n", a);
}

#ifdef _MSC_VER
__declspec (noinline)
#elif defined(__clang__)
__attribute__((noinline))
#endif
void pop(unsigned int a)
{
    printf("pop %d\n", a);
}

template <typename T>
T ror(T l, int y)
{
#ifdef _MSC_VER
    if constexpr (sizeof(T) == 8)      return _rotr64(l, y);
    else if constexpr (sizeof(T) <= 4) return _rotr(l, y);
#elif defined(__clang__)
    if constexpr (sizeof(T) == 8)      return __builtin_rotateright64(l, y);
    else if constexpr (sizeof(T) == 4) return __builtin_rotateright32(l, y);
    else if constexpr (sizeof(T) == 2) return __builtin_rotateright16(l, y);
    else if constexpr (sizeof(T) == 1)  return __builtin_rotateright8(l, y);
#else
#error eh what compiler tho
#endif

    __builtin_unreachable();
}

template <typename T>
T rol(T l, int y)
{
#ifdef _MSC_VER
    if constexpr (sizeof(T) == 8)      return _rotl64(l, y);
    else if constexpr (sizeof(T) <= 4) return _rotl(l, y);
#elif defined(__clang__)
    if constexpr (sizeof(T) == 8)      return __builtin_rotatleft64(l, y);
    else if constexpr (sizeof(T) == 4) return __builtin_rotatleft32(l, y);
    else if constexpr (sizeof(T) == 2) return __builtin_rotatleft16(l, y);
    else if constexpr (sizeof(T) == 1) return __builtin_rotatleft8(l, y);
#else
#error eh what compiler tho
#endif

    __builtin_unreachable();
}

void vm_entry(unsigned int* _ebx, unsigned int* _esi, unsigned int* _edi, unsigned int* _esp, unsigned int* _ebp)
{
#define decode_byte() \
    esi += 1; \
    al ^= bl; \
    al -= 0x3a; \
    al = ror(al, 1); \
    al = -al; \
    al = ~al; \
    bl ^= al;

#define handler_chain(pcode) \
    eax = pcode; \
    esi += 4; \
    eax ^= ebx; \
    eax = ror(eax, 1); \
    eax ^= 0x4acb3db9; \
    eax -= 0x458c0140; \
    eax = rol(eax, 1); \
    ebx ^= eax; \
    edi += eax;

    union
    {
        unsigned int eax;
        struct
        {
            unsigned short ax;
        };
        struct
        {
            unsigned char al;
            unsigned char ah;
        };
    };
    union
    {
        unsigned int ebx;
        struct
        {
            unsigned short bx;
        };
        struct
        {
            unsigned char bl;
            unsigned char bh;
        };
    };

    eax = 0;

    unsigned int esi = 0x9486cfea;
    esi += 0x55106798;
    esi = -esi;
    esi += 0x69733a52;
    esi = rol(esi, 1);
    esi = ~esi;
    esi += eax;

    // mov ebx, esi
    ebx = esi;

    // mov eax, 0
    eax = 0;

    // sub ebx, eax
    ebx -= eax;

    // lea edi, [0x4620e6]
    unsigned int edi = 0x4620e6;
    handler_chain(0x1ECBF564);
    printf("handler_chain\n");

    *_ebx = ebx;
    *_esi = esi;
    *_edi = edi;

    // 4892af
    printf("%08X %08X %08X\n", ebx, esi, edi);

    eax = 0x00000002;
    decode_byte();
    pop(eax);
    handler_chain(0x1ECB5F40);


    *_ebx = ebx;
    *_esi = esi;
    *_edi = edi;


    // 493fb7
    eax = 0x3E;
    decode_byte();
    pop(eax);
    handler_chain(0x1EC1CC61);


    *_ebx = ebx;
    *_esi = esi;
    *_edi = edi;
}

void clang_test()
{
    unsigned int ebx, esi, edi, esp, ebp;
    vm_entry(&ebx, &esi, &edi, &esp, &ebp);
}