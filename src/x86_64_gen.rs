
use std::collections::HashMap;

use definitions::{RegDefinition, SubRegDefinition, Definition, GenDefinition, Endianness};

//pub struct X86;

impl GenDefinition for X86_64 {
    fn gen_definition() -> Definition {
        let mut regs = HashMap::new();
        let mut sub_regs = HashMap::new();
        
    regs.insert("RIP", RegDefinition {
        offset: 656, width: 64, fp: false, ip: true,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec![]
    });
    

    regs.insert("MXCSR", RegDefinition {
        offset: 648, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: true,
        sp: false,
        sub_regs: vec!["IE","DE","ZE","OE","UE","PE","DAZ","IM","DM","ZM","OM","UM","PM","R_NEG","R_POS","FZ"]
    });
    

    sub_regs.insert("IE", SubRegDefinition {
        parent: "MXCSR", from_bit: 0,
        width: 1
    });


    sub_regs.insert("DE", SubRegDefinition {
        parent: "MXCSR", from_bit: 1,
        width: 1
    });


    sub_regs.insert("ZE", SubRegDefinition {
        parent: "MXCSR", from_bit: 2,
        width: 1
    });


    sub_regs.insert("OE", SubRegDefinition {
        parent: "MXCSR", from_bit: 3,
        width: 1
    });


    sub_regs.insert("UE", SubRegDefinition {
        parent: "MXCSR", from_bit: 4,
        width: 1
    });


    sub_regs.insert("PE", SubRegDefinition {
        parent: "MXCSR", from_bit: 5,
        width: 1
    });


    sub_regs.insert("DAZ", SubRegDefinition {
        parent: "MXCSR", from_bit: 6,
        width: 1
    });


    sub_regs.insert("IM", SubRegDefinition {
        parent: "MXCSR", from_bit: 7,
        width: 1
    });


    sub_regs.insert("DM", SubRegDefinition {
        parent: "MXCSR", from_bit: 8,
        width: 1
    });


    sub_regs.insert("ZM", SubRegDefinition {
        parent: "MXCSR", from_bit: 9,
        width: 1
    });


    sub_regs.insert("OM", SubRegDefinition {
        parent: "MXCSR", from_bit: 10,
        width: 1
    });


    sub_regs.insert("UM", SubRegDefinition {
        parent: "MXCSR", from_bit: 11,
        width: 1
    });


    sub_regs.insert("PM", SubRegDefinition {
        parent: "MXCSR", from_bit: 12,
        width: 1
    });


    sub_regs.insert("R_NEG", SubRegDefinition {
        parent: "MXCSR", from_bit: 13,
        width: 1
    });


    sub_regs.insert("R_POS", SubRegDefinition {
        parent: "MXCSR", from_bit: 14,
        width: 1
    });


    sub_regs.insert("FZ", SubRegDefinition {
        parent: "MXCSR", from_bit: 15,
        width: 1
    });


    regs.insert("RFLAGS", RegDefinition {
        offset: 640, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: true,
        sp: false,
        sub_regs: vec!["OF","DF","SF","ZF","AF","PF","CF"]
    });
    

    sub_regs.insert("OF", SubRegDefinition {
        parent: "RFLAGS", from_bit: 11,
        width: 1
    });


    sub_regs.insert("DF", SubRegDefinition {
        parent: "RFLAGS", from_bit: 10,
        width: 1
    });


    sub_regs.insert("SF", SubRegDefinition {
        parent: "RFLAGS", from_bit: 7,
        width: 1
    });


    sub_regs.insert("ZF", SubRegDefinition {
        parent: "RFLAGS", from_bit: 6,
        width: 1
    });


    sub_regs.insert("AF", SubRegDefinition {
        parent: "RFLAGS", from_bit: 4,
        width: 1
    });


    sub_regs.insert("PF", SubRegDefinition {
        parent: "RFLAGS", from_bit: 2,
        width: 1
    });


    sub_regs.insert("CF", SubRegDefinition {
        parent: "RFLAGS", from_bit: 0,
        width: 1
    });


    regs.insert("YMM15", RegDefinition {
        offset: 608, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM15"]
    });
    

    sub_regs.insert("XMM15", SubRegDefinition {
        parent: "YMM15", from_bit: 0,
        width: 128
    });


    regs.insert("YMM14", RegDefinition {
        offset: 576, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM14"]
    });
    

    sub_regs.insert("XMM14", SubRegDefinition {
        parent: "YMM14", from_bit: 0,
        width: 128
    });


    regs.insert("YMM13", RegDefinition {
        offset: 544, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM13"]
    });
    

    sub_regs.insert("XMM13", SubRegDefinition {
        parent: "YMM13", from_bit: 0,
        width: 128
    });


    regs.insert("YMM12", RegDefinition {
        offset: 512, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM12"]
    });
    

    sub_regs.insert("XMM12", SubRegDefinition {
        parent: "YMM12", from_bit: 0,
        width: 128
    });


    regs.insert("YMM11", RegDefinition {
        offset: 480, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM11"]
    });
    

    sub_regs.insert("XMM11", SubRegDefinition {
        parent: "YMM11", from_bit: 0,
        width: 128
    });


    regs.insert("YMM10", RegDefinition {
        offset: 448, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM10"]
    });
    

    sub_regs.insert("XMM10", SubRegDefinition {
        parent: "YMM10", from_bit: 0,
        width: 128
    });


    regs.insert("YMM9", RegDefinition {
        offset: 416, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM9"]
    });
    

    sub_regs.insert("XMM9", SubRegDefinition {
        parent: "YMM9", from_bit: 0,
        width: 128
    });


    regs.insert("YMM8", RegDefinition {
        offset: 384, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM8"]
    });
    

    sub_regs.insert("XMM8", SubRegDefinition {
        parent: "YMM8", from_bit: 0,
        width: 128
    });


    regs.insert("YMM7", RegDefinition {
        offset: 352, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM7"]
    });
    

    sub_regs.insert("XMM7", SubRegDefinition {
        parent: "YMM7", from_bit: 0,
        width: 128
    });


    regs.insert("YMM6", RegDefinition {
        offset: 320, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM6"]
    });
    

    sub_regs.insert("XMM6", SubRegDefinition {
        parent: "YMM6", from_bit: 0,
        width: 128
    });


    regs.insert("YMM5", RegDefinition {
        offset: 288, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM5"]
    });
    

    sub_regs.insert("XMM5", SubRegDefinition {
        parent: "YMM5", from_bit: 0,
        width: 128
    });


    regs.insert("YMM4", RegDefinition {
        offset: 256, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM4"]
    });
    

    sub_regs.insert("XMM4", SubRegDefinition {
        parent: "YMM4", from_bit: 0,
        width: 128
    });


    regs.insert("YMM3", RegDefinition {
        offset: 224, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM3"]
    });
    

    sub_regs.insert("XMM3", SubRegDefinition {
        parent: "YMM3", from_bit: 0,
        width: 128
    });


    regs.insert("YMM2", RegDefinition {
        offset: 192, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM2"]
    });
    

    sub_regs.insert("XMM2", SubRegDefinition {
        parent: "YMM2", from_bit: 0,
        width: 128
    });


    regs.insert("YMM1", RegDefinition {
        offset: 160, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM1"]
    });
    

    sub_regs.insert("XMM1", SubRegDefinition {
        parent: "YMM1", from_bit: 0,
        width: 128
    });


    regs.insert("YMM0", RegDefinition {
        offset: 128, width: 256, fp: true, ip: false,
        vector: true, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["XMM0"]
    });
    

    sub_regs.insert("XMM0", SubRegDefinition {
        parent: "YMM0", from_bit: 0,
        width: 128
    });


    regs.insert("RSP", RegDefinition {
        offset: 120, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: true,
        sub_regs: vec!["SPL","SP","ESP"]
    });
    

    sub_regs.insert("SPL", SubRegDefinition {
        parent: "RSP", from_bit: 0,
        width: 8
    });


    sub_regs.insert("SP", SubRegDefinition {
        parent: "RSP", from_bit: 0,
        width: 16
    });


    sub_regs.insert("ESP", SubRegDefinition {
        parent: "RSP", from_bit: 0,
        width: 32
    });


    regs.insert("RBP", RegDefinition {
        offset: 112, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: true,
        sub_regs: vec!["BPL","BP","EBP"]
    });
    

    sub_regs.insert("BPL", SubRegDefinition {
        parent: "RBP", from_bit: 0,
        width: 8
    });


    sub_regs.insert("BP", SubRegDefinition {
        parent: "RBP", from_bit: 0,
        width: 16
    });


    sub_regs.insert("EBP", SubRegDefinition {
        parent: "RBP", from_bit: 0,
        width: 32
    });


    regs.insert("R15", RegDefinition {
        offset: 104, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["R15B","R15W","R15D"]
    });
    

    sub_regs.insert("R15B", SubRegDefinition {
        parent: "R15", from_bit: 0,
        width: 8
    });


    sub_regs.insert("R15W", SubRegDefinition {
        parent: "R15", from_bit: 0,
        width: 16
    });


    sub_regs.insert("R15D", SubRegDefinition {
        parent: "R15", from_bit: 0,
        width: 32
    });


    regs.insert("R14", RegDefinition {
        offset: 96, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["R14B","R14W","R14D"]
    });
    

    sub_regs.insert("R14B", SubRegDefinition {
        parent: "R14", from_bit: 0,
        width: 8
    });


    sub_regs.insert("R14W", SubRegDefinition {
        parent: "R14", from_bit: 0,
        width: 16
    });


    sub_regs.insert("R14D", SubRegDefinition {
        parent: "R14", from_bit: 0,
        width: 32
    });


    regs.insert("R13", RegDefinition {
        offset: 88, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["R13B","R13W","R13D"]
    });
    

    sub_regs.insert("R13B", SubRegDefinition {
        parent: "R13", from_bit: 0,
        width: 8
    });


    sub_regs.insert("R13W", SubRegDefinition {
        parent: "R13", from_bit: 0,
        width: 16
    });


    sub_regs.insert("R13D", SubRegDefinition {
        parent: "R13", from_bit: 0,
        width: 32
    });


    regs.insert("R12", RegDefinition {
        offset: 80, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["R12B","R12W","R12D"]
    });
    

    sub_regs.insert("R12B", SubRegDefinition {
        parent: "R12", from_bit: 0,
        width: 8
    });


    sub_regs.insert("R12W", SubRegDefinition {
        parent: "R12", from_bit: 0,
        width: 16
    });


    sub_regs.insert("R12D", SubRegDefinition {
        parent: "R12", from_bit: 0,
        width: 32
    });


    regs.insert("R11", RegDefinition {
        offset: 72, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["R11B","R11W","R11D"]
    });
    

    sub_regs.insert("R11B", SubRegDefinition {
        parent: "R11", from_bit: 0,
        width: 8
    });


    sub_regs.insert("R11W", SubRegDefinition {
        parent: "R11", from_bit: 0,
        width: 16
    });


    sub_regs.insert("R11D", SubRegDefinition {
        parent: "R11", from_bit: 0,
        width: 32
    });


    regs.insert("R10", RegDefinition {
        offset: 64, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["R10B","R10W","R10D"]
    });
    

    sub_regs.insert("R10B", SubRegDefinition {
        parent: "R10", from_bit: 0,
        width: 8
    });


    sub_regs.insert("R10W", SubRegDefinition {
        parent: "R10", from_bit: 0,
        width: 16
    });


    sub_regs.insert("R10D", SubRegDefinition {
        parent: "R10", from_bit: 0,
        width: 32
    });


    regs.insert("R9", RegDefinition {
        offset: 56, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["R9B","R9W","R9D"]
    });
    

    sub_regs.insert("R9B", SubRegDefinition {
        parent: "R9", from_bit: 0,
        width: 8
    });


    sub_regs.insert("R9W", SubRegDefinition {
        parent: "R9", from_bit: 0,
        width: 16
    });


    sub_regs.insert("R9D", SubRegDefinition {
        parent: "R9", from_bit: 0,
        width: 32
    });


    regs.insert("R8", RegDefinition {
        offset: 48, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["R8B","R8W","R8D"]
    });
    

    sub_regs.insert("R8B", SubRegDefinition {
        parent: "R8", from_bit: 0,
        width: 8
    });


    sub_regs.insert("R8W", SubRegDefinition {
        parent: "R8", from_bit: 0,
        width: 16
    });


    sub_regs.insert("R8D", SubRegDefinition {
        parent: "R8", from_bit: 0,
        width: 32
    });


    regs.insert("RDI", RegDefinition {
        offset: 40, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["DIL","DI","EDI"]
    });
    

    sub_regs.insert("DIL", SubRegDefinition {
        parent: "RDI", from_bit: 0,
        width: 8
    });


    sub_regs.insert("DI", SubRegDefinition {
        parent: "RDI", from_bit: 0,
        width: 16
    });


    sub_regs.insert("EDI", SubRegDefinition {
        parent: "RDI", from_bit: 0,
        width: 32
    });


    regs.insert("RSI", RegDefinition {
        offset: 32, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["SIL","SI","ESI"]
    });
    

    sub_regs.insert("SIL", SubRegDefinition {
        parent: "RSI", from_bit: 0,
        width: 8
    });


    sub_regs.insert("SI", SubRegDefinition {
        parent: "RSI", from_bit: 0,
        width: 16
    });


    sub_regs.insert("ESI", SubRegDefinition {
        parent: "RSI", from_bit: 0,
        width: 32
    });


    regs.insert("RDX", RegDefinition {
        offset: 24, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["DL","DX","EDX"]
    });
    

    sub_regs.insert("DL", SubRegDefinition {
        parent: "RDX", from_bit: 0,
        width: 8
    });


    sub_regs.insert("DX", SubRegDefinition {
        parent: "RDX", from_bit: 0,
        width: 16
    });


    sub_regs.insert("EDX", SubRegDefinition {
        parent: "RDX", from_bit: 0,
        width: 32
    });


    regs.insert("RCX", RegDefinition {
        offset: 16, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["CL","CX","ECX"]
    });
    

    sub_regs.insert("CL", SubRegDefinition {
        parent: "RCX", from_bit: 0,
        width: 8
    });


    sub_regs.insert("CX", SubRegDefinition {
        parent: "RCX", from_bit: 0,
        width: 16
    });


    sub_regs.insert("ECX", SubRegDefinition {
        parent: "RCX", from_bit: 0,
        width: 32
    });


    regs.insert("RBX", RegDefinition {
        offset: 8, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["BL","BX","EBX"]
    });
    

    sub_regs.insert("BL", SubRegDefinition {
        parent: "RBX", from_bit: 0,
        width: 8
    });


    sub_regs.insert("BX", SubRegDefinition {
        parent: "RBX", from_bit: 0,
        width: 16
    });


    sub_regs.insert("EBX", SubRegDefinition {
        parent: "RBX", from_bit: 0,
        width: 32
    });


    regs.insert("RAX", RegDefinition {
        offset: 0, width: 64, fp: false, ip: false,
        vector: false, segment: false, flags: false,
        sp: false,
        sub_regs: vec!["AL","AX","EAX"]
    });
    

    sub_regs.insert("AL", SubRegDefinition {
        parent: "RAX", from_bit: 0,
        width: 8
    });


    sub_regs.insert("AX", SubRegDefinition {
        parent: "RAX", from_bit: 0,
        width: 16
    });


    sub_regs.insert("EAX", SubRegDefinition {
        parent: "RAX", from_bit: 0,
        width: 32
    });

        Definition {
            base_reg: "RAX",
            regs_size: 664,
            regs: regs,
            sub_regs: sub_regs,
            epilogue: "
movabs [ %0], rax
movq rax, %0
stmxcsr [rax+648]
vmovupd [rax+608], YMM15
vmovupd [rax+576], YMM14
vmovupd [rax+544], YMM13
vmovupd [rax+512], YMM12
vmovupd [rax+480], YMM11
vmovupd [rax+448], YMM10
vmovupd [rax+416], YMM9
vmovupd [rax+384], YMM8
vmovupd [rax+352], YMM7
vmovupd [rax+320], YMM6
vmovupd [rax+288], YMM5
vmovupd [rax+256], YMM4
vmovupd [rax+224], YMM3
vmovupd [rax+192], YMM2
vmovupd [rax+160], YMM1
vmovupd [rax+128], YMM0
mov [rax+120], RSP
mov [rax+112], RBP
mov [rax+104], R15
mov [rax+96], R14
mov [rax+88], R13
mov [rax+80], R12
mov [rax+72], R11
mov [rax+64], R10
mov [rax+56], R9
mov [rax+48], R8
mov [rax+40], RDI
mov [rax+32], RSI
mov [rax+24], RDX
mov [rax+16], RCX
mov [rax+8], RBX
lea rsp, [rax+640+8]; pushfq",
            prologue: "
int3; movq rax, %0
lea rsp, [rax+640]; popfq
ldmxcsr [rax+648]
vmovdqu YMM15, [rax+608]
vmovdqu YMM14, [rax+576]
vmovdqu YMM13, [rax+544]
vmovdqu YMM12, [rax+512]
vmovdqu YMM11, [rax+480]
vmovdqu YMM10, [rax+448]
vmovdqu YMM9, [rax+416]
vmovdqu YMM8, [rax+384]
vmovdqu YMM7, [rax+352]
vmovdqu YMM6, [rax+320]
vmovdqu YMM5, [rax+288]
vmovdqu YMM4, [rax+256]
vmovdqu YMM3, [rax+224]
vmovdqu YMM2, [rax+192]
vmovdqu YMM1, [rax+160]
vmovdqu YMM0, [rax+128]
mov RSP, [rax+120]
mov RBP, [rax+112]
mov R15, [rax+104]
mov R14, [rax+96]
mov R13, [rax+88]
mov R12, [rax+80]
mov R11, [rax+72]
mov R10, [rax+64]
mov R9, [rax+56]
mov R8, [rax+48]
mov RDI, [rax+40]
mov RSI, [rax+32]
mov RDX, [rax+24]
mov RCX, [rax+16]
mov RBX, [rax+8]
mov rax, [rax+0]",
            epilogue_steps: 36,
            prologue_steps: 37,
            endian: Endianness::Little
        }
    }
}
