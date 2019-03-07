
MEMORY {

    VECTABLE    : ORIGIN = 0x20000000 	LENGTH = 4K
    ICCM0       : ORIGIN = 0x20001000 	LENGTH = 252K
    DCCM        : ORIGIN = 0x80000000, LENGTH = 0x00020000
    XCCM        : ORIGIN = 0xc0000000, LENGTH = 0x00008000
    YCCM        : ORIGIN = 0xe0000000, LENGTH = 0x00008000
    }


SECTIONS {
    ivt :
        {
            KEEP (*(.ivt));
        }  > VECTABLE
    startup :
        {
            KEEP (*crt0.o(.text.__startup))
        }  > ICCM0 
    .Xdata :
       {
          *(.Xdata*)
       } > XCCM
    .mli_model :
       {
           *(.mli_model)
       } > XCCM
    .Ydata :
       {
          *(.Ydata*)
       } > YCCM
    .Zdata :
       {
          *(.Zdata*)
       } > DCCM
    .mli_model_p2 :
       {
          *(.mli_model_p2)
       } > DCCM
 /* Read-only sections, merged into text segment: */
  .hash          : { *(.hash)		}
  .dynsym        : { *(.dynsym)		}
  .dynstr        : { *(.dynstr)		}
  .gnu.version   : { *(.gnu.version)	}
  .gnu.version_d   : { *(.gnu.version_d)	}
  .gnu.version_r   : { *(.gnu.version_r)	}
  .rel.init       : { *(.rel.init) }
  .rela.init      : { *(.rela.init) }
  .rel.text       : { *(.rel.text .rel.text.* .rel.gnu.linkonce.t.*) }
  .rela.text      : { *(.rela.text .rela.text.* .rela.gnu.linkonce.t.*) }
  .rel.fini       : { *(.rel.fini) }
  .rela.fini      : { *(.rela.fini) }
  .rel.rodata     : { *(.rel.rodata .rel.rodata.* .rel.gnu.linkonce.r.*) }
  .rela.rodata    : { *(.rela.rodata .rela.rodata.* .rela.gnu.linkonce.r.*) }
  .rel.data       : { *(.rel.data .rel.data.* .rel.gnu.linkonce.d.*) }
  .rela.data      : { *(.rela.data .rela.data.* .rela.gnu.linkonce.d.*) }
  .rel.tdata	  : { *(.rel.tdata .rel.tdata.* .rel.gnu.linkonce.td.*) }
  .rela.tdata	  : { *(.rela.tdata .rela.tdata.* .rela.gnu.linkonce.td.*) }
  .rel.tbss	  : { *(.rel.tbss .rel.tbss.* .rel.gnu.linkonce.tb.*) }
  .rela.tbss	  : { *(.rela.tbss .rela.tbss.* .rela.gnu.linkonce.tb.*) }
  .rel.ctors      : { *(.rel.ctors) }
  .rela.ctors     : { *(.rela.ctors) }
  .rel.dtors      : { *(.rel.dtors) }
  .rela.dtors     : { *(.rela.dtors) }
  .rel.got        : { *(.rel.got) }
  .rela.got       : { *(.rela.got) }
  .rel.sdata      : { *(.rel.sdata .rel.sdata.* .rel.gnu.linkonce.s.*) }
  .rela.sdata     : { *(.rela.sdata .rela.sdata.* .rela.gnu.linkonce.s.*) }
  .rel.sbss       : { *(.rel.sbss .rel.sbss.* .rel.gnu.linkonce.sb.*) }
  .rela.sbss      : { *(.rela.sbss .rela.sbss.* .rela.gnu.linkonce.sb.*) }
  .rel.sdata2     : { *(.rel.sdata2 .rel.sdata2.* .rel.gnu.linkonce.s2.*) }
  .rela.sdata2    : { *(.rela.sdata2 .rela.sdata2.* .rela.gnu.linkonce.s2.*) }
  .rel.sbss2      : { *(.rel.sbss2 .rel.sbss2.* .rel.gnu.linkonce.sb2.*) }
  .rela.sbss2     : { *(.rela.sbss2 .rela.sbss2.* .rela.gnu.linkonce.sb2.*) }
  .rel.bss        : { *(.rel.bss .rel.bss.* .rel.gnu.linkonce.b.*) }
  .rela.bss       : { *(.rela.bss .rela.bss.* .rela.gnu.linkonce.b.*) }
  .jcr : { KEEP (*(.jcr)) } > ICCM0
  .eh_frame : { KEEP (*(.eh_frame)) } > ICCM0
  .gcc_except_table : { *(.gcc_except_table) *(.gcc_except_table.*) } > ICCM0
  .plt : { *(.plt) } > ICCM0
  .jlitab :
  {
    __JLI_TABLE__ = .;
     jlitab*.o:(.jlitab*) *(.jlitab*)
  } > ICCM0
  .rodata   :
  {
    *(.rodata) *(.rodata.*) *(.gnu.linkonce.r.*)
  } > ICCM0
  .rodata1        : { *(.rodata1) } > ICCM0
  .init           :
  {
    KEEP (*(.init))
  }  > ICCM0  =0
  .text           :
  {
    *(.text .stub .text.* .gnu.linkonce.t.*)
    /* .gnu.warning sections are handled specially by elf32.em.  */
    *(.gnu.warning)
  }  > ICCM0 =0
  .fini           :
  {
    KEEP (*(.fini))
    PROVIDE (__etext = .);
    PROVIDE (_etext = .);
    PROVIDE (etext = .);
  }  > ICCM0 =0
  /* Start of the data section image in ROM.  */
  __data_image = .;
  PROVIDE (__data_image = .);
  .data	  :
  {
     PROVIDE (__data_start = .) ;
    /* --gc-sections will delete empty .data. This leads to wrong start
       addresses for subsequent sections because -Tdata= from the command
       line will have no effect, see PR13697.  Thus, keep .data  */
    KEEP (*(.data))
    *(.data .data.* .gnu.linkonce.d.*)
    SORT(CONSTRUCTORS)
  }  > DCCM
  .got            : { *(.got.plt) *(.got) }  > DCCM
  .ctors          :
  {
    /* gcc uses crtbegin.o to find the start of
       the constructors, so we make sure it is
       first.  Because this is a wildcard, it
       doesn't matter if the user does not
       actually link against crtbegin.o; the
       linker won't look for a file to match a
       wildcard.  The wildcard also means that it
       doesn't matter which directory crtbegin.o
       is in.  */
    KEEP (*crtbegin*.o(.ctors))
    /* We don't want to include the .ctor section from
       from the crtend.o file until after the sorted ctors.
       The .ctor section from the crtend file contains the
       end of ctors marker and it must be last */
    KEEP (*(EXCLUDE_FILE (*crtend*.o ) .ctors))
    KEEP (*(SORT(.ctors.*)))
    KEEP (*(.ctors))
  }  > DCCM
  .dtors          :
  {
    KEEP (*crtbegin*.o(.dtors))
    KEEP (*(EXCLUDE_FILE (*crtend*.o ) .dtors))
    KEEP (*(SORT(.dtors.*)))
    KEEP (*(.dtors))
  }  > DCCM
  /* We want the small data sections together, so single-instruction offsets
     can access them all, and initialized data all before uninitialized, so
     we can shorten the on-disk segment size.  */
  .sdata          :
  {
    __SDATA_BEGIN__ = . + 0x100;
    *(.sdata .sdata.* .gnu.linkonce.s.*)
    _edata  =  .;
    PROVIDE (edata = .);
  }  > DCCM
  .sdata2         : { *(.sdata2 .sdata2.* .gnu.linkonce.s2.*) }  > DCCM
  .sbss           :
  {
    PROVIDE (__sbss_start = .);
    PROVIDE (___sbss_start = .);
    *(.dynsbss)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
    PROVIDE (__sbss_end = .);
    PROVIDE (___sbss_end = .);
  }  > DCCM
  .sbss2          : { *(.sbss2 .sbss2.* .gnu.linkonce.sb2.*) }  > DCCM
  .bss            :
  {
   *(.dynbss)
   *(.bss .bss.* .gnu.linkonce.b.*)
   *(COMMON)
   /* Align here to ensure that the .bss section occupies space up to
      _end.  Align after .bss to ensure correct alignment even if the
      .bss section disappears because there are no input sections.  */
   . = ALIGN(32 / 8);
   _end = .;
   PROVIDE (end = .);
  }  > DCCM
  /* Global data not cleared after reset.  */
  .noinit  :
  {
    *(.noinit*)
    . = ALIGN(32 / 8);
     PROVIDE (__start_heap = .) ;
  }  > DCCM
  /* Stabs debugging sections.  */
  .stab          0 : { *(.stab) }
  .stabstr       0 : { *(.stabstr) }
  .stab.excl     0 : { *(.stab.excl) }
  .stab.exclstr  0 : { *(.stab.exclstr) }
  .stab.index    0 : { *(.stab.index) }
  .stab.indexstr 0 : { *(.stab.indexstr) }
  .comment       0 : { *(.comment) }
  /* DWARF debug sections.
     Symbols in the DWARF debugging sections are relative to the beginning
     of the section so we begin them at 0.  */
  /* DWARF 1 */
  .debug          0 : { *(.debug) }
  .line           0 : { *(.line) }
  /* GNU DWARF 1 extensions */
  .debug_srcinfo  0 : { *(.debug_srcinfo) }
  .debug_sfnames  0 : { *(.debug_sfnames) }
  /* DWARF 1.1 and DWARF 2 */
  .debug_aranges  0 : { *(.debug_aranges) }
  .debug_pubnames 0 : { *(.debug_pubnames) }
  /* DWARF 2 */
  .debug_info     0 : { *(.debug_info) *(.gnu.linkonce.wi.*) }
  .debug_abbrev   0 : { *(.debug_abbrev) }
  .debug_line     0 : { *(.debug_line) }
  .debug_frame    0 : { *(.debug_frame) }
  .debug_str      0 : { *(.debug_str) }
  .debug_loc      0 : { *(.debug_loc) }
  .debug_macinfo  0 : { *(.debug_macinfo) }
  /* DWARF 3 */
  .debug_pubtypes 0 : { *(.debug_pubtypes) }
  .debug_ranges   0 : { *(.debug_ranges) }
  /* DWARF Extension.  */
  .debug_macro    0 : { *(.debug_macro) }
  /* ARC Extension Sections */
  .arcextmap	  0 : { *(.gnu.linkonce.arcextmap.*) } 
}

REGION_ALIAS("startup", ICCM0)
REGION_ALIAS("text", ICCM0)
REGION_ALIAS("data", DCCM)
REGION_ALIAS("sdata", DCCM)


PROVIDE (__stack_top = (0x8001ffff & -4 ));
PROVIDE (__end_heap =  (0x8001ffff ));
