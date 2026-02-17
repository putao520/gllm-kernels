	.file	"rust_out.e7f190b46a0de160-cgu.0"
	.section	.text._RNvCsjUD5jPx6gGO_8rust_out8test_amx,"ax",@progbits
	.globl	_RNvCsjUD5jPx6gGO_8rust_out8test_amx
	.p2align	4
	.type	_RNvCsjUD5jPx6gGO_8rust_out8test_amx,@function
_RNvCsjUD5jPx6gGO_8rust_out8test_amx:
	.cfi_startproc
	xorl	%eax, %eax
	movl	$64, %ecx
	tileloadd	(%rax,%rcx), %tmm0
	retq
.Lfunc_end0:
	.size	_RNvCsjUD5jPx6gGO_8rust_out8test_amx, .Lfunc_end0-_RNvCsjUD5jPx6gGO_8rust_out8test_amx
	.cfi_endproc

	.ident	"rustc version 1.95.0-nightly (a33907a7a 2026-02-14)"
	.section	".note.GNU-stack","",@progbits
