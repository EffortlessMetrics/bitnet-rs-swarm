L0:
(W)     mov (8|M0)               r127.0<1>:ud  0x0:ud
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud
(W)     and (1|M0)               r127.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x40:ud              {I@2}
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}
(W)     mad (1|M0)               r127.2<1>:ud  r127.2<0;0>:ud    r127.0<0;0>:uw    0xC0:uw              {I@1}
(W)     mov (8|M0)               r7.0<1>:ud    r1.0<1;1,0>:ud
(W)     send.dc0 (8|M0)          r1       r127    null:0  0x0            0x024844FD           {A@1,$0} // wr:1h+0, rd:4; oword aligned block read x8
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x80:uw              {$0.src}
(W)     send.dc0 (8|M0)          r5       r127    null:0  0x0            0x022843FD           {A@1,$1} // wr:1h+0, rd:2; oword aligned block read x4
        nop
        nop
(W)     mov (8|M0)               r127.0<1>:ud  0x0:ud                              {$1.src}
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}
(W)     send.dc0 (8|M0)          r8       r127    null:0  0x0            0x022843FD           {A@1,$2} // wr:1h+0, rd:2; oword aligned block read x4
(W)     mov (8|M0)               r3.0<1>:ud    r0.0<1;1,0>:ud                   {$0.dst}
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {A@1}
(W)     mul (1|M0)               acc0.0<1>:d   r7.3<0;1,0>:d     r3.2<0;1,0>:uw   {A@1}
(W)     mach (1|M0)              r4.0<1>:d     r7.3<0;1,0>:d     r3.1<0;1,0>:d
        sync.nop                             null                             {Compacted,$1.dst}
        add3 (16|M0)             r5.0<1>:d     r4.0<0;0>:d       r1.0<1;0>:uw      r7.0<0>:d        {I@1}
        add3 (16|M16)            r10.0<1>:d    r4.0<0;0>:d       r2.0<1;0>:uw      r7.0<0>:d
        sync.nop                             null                             {Compacted,$2.dst}
        cmp (16|M0)   (lt)f1.0   null<1>:d     r5.0<1;1,0>:ud    r8.4<0;1,0>:ud   {I@2}
        cmp (16|M16)  (lt)f1.0   null<1>:d     r10.0<1;1,0>:ud   r8.4<0;1,0>:ud   {I@2}
(~f1.0) goto (32|M0)                         L1584                  L1584
L416:
(W)     cmp (16|M0)   (eq)f0.0   null<1>:d     r8.5<0;1,0>:d     0:w               {Compacted}
(W)     cmp (16|M16)  (eq)f0.0   null<1>:d     r8.5<0;1,0>:d     0:w
(~f0.0) goto (32|M0)                         L504                  L504
L456:
        mov (16|M0)              r12.0<1>:d    0:w
        mov (16|M16)             r14.0<1>:d    0:w
        goto (32|M0)                         L504                  L1144
L504:
        join (32|M0)                         L1144
L520:
(W)     mul (8|M0)               acc0.0<1>:d   r5.0<1;1,0>:d     r8.12<0;1,0>:uw  {Compacted}
        mach (8|M0)              r16.0<1>:d    r5.0<1;1,0>:d     r8.6<0;1,0>:d    {Compacted}
(W)     mul (8|M8)               acc0.0<1>:d   r6.0<1;1,0>:d     r8.12<0;1,0>:uw
        mach (8|M8)              r17.0<1>:d    r6.0<1;1,0>:d     r8.6<0;1,0>:d    {Compacted}
(W)     mul (8|M16)              acc0.0<1>:d   r10.0<1;1,0>:d    r8.12<0;1,0>:uw
        mov (16|M0)              r12.0<1>:d    0:w
        mov (16|M16)             r14.0<1>:d    0:w
        mach (8|M16)             r18.0<1>:d    r10.0<1;1,0>:d    r8.6<0;1,0>:d    {Compacted}
(W)     mul (8|M24)              acc0.0<1>:d   r11.0<1;1,0>:d    r8.12<0;1,0>:uw
(W)     mov (1|M0)               r4.1<1>:d     0:w
        mach (8|M24)             r19.0<1>:d    r11.0<1;1,0>:d    r8.6<0;1,0>:d    {Compacted}
L656:
(W)     shr (1|M0)               r4.6<1>:ud    r4.1<0;1,0>:ud    2:w               {I@2}
(W)     and (1|M0)               r4.5<1>:d     r4.1<0;1,0>:d     31:w
(W)     and (1|M0)               r4.7<1>:d     r4.6<0;1,0>:d     1073741760:d               {I@2}
(W)     and (1|M0)               r7.6<1>:d     r4.6<0;1,0>:d     32:w
(W)     and (1|M0)               r4.2<1>:d     r4.1<0;1,0>:d     255:w
        add3 (16|M0)             r20.0<1>:d    r16.0<1;0>:d      r4.7<0;0>:d       r7.6<0>:d        {I@2}
        add3 (16|M16)            r22.0<1>:d    r18.0<1;0>:d      r4.7<0;0>:d       r7.6<0>:d
        add3 (16|M0)             r24.0<1>:d    r20.0<1;0>:d      r4.5<0;0>:d       r9.3<0>:d        {I@2}
        add3 (16|M16)            r26.0<1>:d    r22.0<1;0>:d      r4.5<0;0>:d       r9.3<0>:d        {I@2}
        send.ugm (16|M0)         r28      r24     null:0  0x1000000            0x64280900           {A@2,$3} // wr:2+0, rd:2; load.ugm.d8u32.a32.ca.ca.bti[1]
        send.ugm (16|M16)        r30      r26     null:0  0x1000000            0x64280900           {A@1,$4} // wr:2+0, rd:2; load.ugm.d8u32.a32.ca.ca.bti[1]
(W)     and (1|M0)               r4.3<1>:d     r4.1<0;1,0>:d     128:w
(W)     add (1|M0)               r36.0<1>:d    r4.1<0;1,0>:d     r9.2<0;1,0>:d
(W)     add (1|M0)               r4.4<1>:d     r4.2<0;1,0>:d     -r4.3<0;1,0>:d   {I@2}
(W)     send.ugm (1|M0)          r37      r36     null:0  0x0            0x62180900           {A@2,$5} // wr:1+0, rd:1; load.ugm.d8u32.a32.ca.ca.bti[0]
(W)     shr (1|M0)               r7.7<1>:ud    r4.4<0;1,0>:ud    4:w               {I@1}
(W)     add (1|M0)               r4.1<1>:d     r4.1<0;1,0>:d     1:w               {Compacted}
(W)     and (1|M0)               r8.0<1>:d     r7.7<0;1,0>:d     30:w               {Compacted,I@2}
        mov (16|M0)              r32.0<1>:d    r28.0<4;1,0>:ub                  {$3.dst}
(W)     add (1|M0)               r8.1<1>:d     -r8.0<0;1,0>:d    6:w               {Compacted,I@2}
        mov (16|M16)             r34.0<1>:d    r30.0<4;1,0>:ub                  {$4.dst}
(W)     and (1|M0)               r8.2<1>:d     r8.1<0;1,0>:d     30:w               {I@2}
(W)     cmp (16|M0)   (lt)f1.0   null<1>:d     r4.1<0;1,0>:ud    r8.5<0;1,0>:ud
        shr (16|M0)              r32.0<1>:ud   r32.0<1;1,0>:ud   r8.2<0;1,0>:d    {I@2}
        shr (16|M16)             r34.0<1>:ud   r34.0<1;1,0>:ud   r8.2<0;1,0>:d
(W)     cmp (16|M16)  (lt)f1.0   null<1>:d     r4.1<0;1,0>:ud    r8.5<0;1,0>:ud
(W)     mov (1|M0)               r36.4<1>:w    r37.0<0;1,0>:b                   {$5.dst}
        and (16|M0)              r32.0<1>:d    r32.0<1;1,0>:d    3:w               {Compacted,I@4}
        and (16|M16)             r34.0<1>:d    r34.0<1;1,0>:d    3:w               {Compacted,I@4}
        mul (16|M0)              r32.0<1>:d    r32.0<1;1,0>:d    r36.4<0;1,0>:w   {I@2}
        mul (16|M16)             r34.0<1>:d    r34.0<1;1,0>:d    r36.4<0;1,0>:w   {I@2}
        add (16|M0)              r12.0<1>:d    r12.0<1;1,0>:d    r32.0<1;1,0>:d   {Compacted,I@2}
        add (16|M16)             r14.0<1>:d    r14.0<1;1,0>:d    r34.0<1;1,0>:d   {Compacted,I@2}
(f1.0)  goto.b (32|M0)                       L1144                  L656
L1144:
        join (32|M0)                         L1584
L1160:
(W)     and (1|M0)    (eq)f0.0   r8.3<1>:d     r9.0<0;1,0>:d     2139095040:d
(W)     mov (1|M0)               r36.3<1>:ud   0x4F800000:ud
(W)     and (1|M0)               r36.1<1>:d    r9.0<0;1,0>:d     8388607:d
        add (16|M0)              r38.0<1>:d    r12.0<1;1,0>:d    -r8.7<0;1,0>:d   {Compacted,I@7}
(W&f0.0) sel (1|M0)              r9.4<1>:f     r36.3<0;1,0>:f    1.0:f               {I@3}
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r36.1<0;1,0>:d    0:w               {I@2}
(W)     cmp (16|M16)  (eq)f1.0   null<1>:d     r36.1<0;1,0>:d    0:w
(W)     cmp (1|M0)    (ge)f0.0   null<1>:ud    r8.3<0;1,0>:ud    0x64000000:ud              {F@1}
(W&~f1.0) cmp (16|M0) (eq)f1.0   null<1>:d     r8.3<0;1,0>:d     0:w
(W&~f1.0) cmp (16|M16) (eq)f1.0  null<1>:d     r8.3<0;1,0>:d     0:w
(W&~f0.0) sel (1|M0)             r9.5<1>:f     r9.4<0;1,0>:f     0x2F800000:f
        add (16|M16)             r40.0<1>:d    r14.0<1;1,0>:d    -r8.7<0;1,0>:d   {Compacted}
(W)     mul (1|M0)               r9.6<1>:f     r9.0<0;1,0>:f     r9.5<0;1,0>:f    {F@1}
(W)     not (1|M0)               f0.0<1>:uw    f1.0<0;1,0>:uw                   {Compacted}
(W)     not (1|M16)              f0.1<1>:uw    f1.1<0;1,0>:uw
        sync.nop                             null                             {Compacted,F@1}
(W)     math.inv (1|M0)          r9.7<1>:f     r9.6<0;1,0>:f                    {$6}
        mov (16|M0)              r42.0<1>:f    r38.0<1;1,0>:d                   {Compacted,I@7}
        mov (16|M16)             r44.0<1>:f    r40.0<1;1,0>:d                   {Compacted,I@3}
        sync.nop                             null                             {Compacted,$6.dst}
        mul (16|M0)              acc0.0<1>:f   r9.7<0;1,0>:f     r42.0<1;1,0>:f   {Compacted,F@2}
(f0.0)  cmp (16|M0)   (eq)f0.0   null<1>:f     r42.0<1;1,0>:f    r9.0<0;1,0>:f    {I@1}
        mul (16|M16)             acc2.0<1>:f   r9.7<0;1,0>:f     r44.0<1;1,0>:f   {Compacted,F@3}
(f0.0)  cmp (16|M16)  (eq)f0.0   null<1>:f     r44.0<1;1,0>:f    r9.0<0;1,0>:f
        mul (16|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f   r9.5<0;1,0>:f    {Compacted}
        mul (16|M16)             acc2.0<1>:f   acc2.0<1;1,0>:f   r9.5<0;1,0>:f    {Compacted}
(~f0.0) sel (16|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f   1.0:f
(~f0.0) sel (16|M16)             acc2.0<1>:f   acc2.0<1;1,0>:f   1.0:f
        shl (16|M0)              r58.0<1>:d    r5.0<1;1,0>:d     2:w               {Compacted}
        shl (16|M16)             r60.0<1>:d    r10.0<1;1,0>:d    2:w               {Compacted}
        mul (16|M0)              r54.0<1>:f    acc0.0<1;1,0>:f   r9.1<0;1,0>:f    {Compacted}
        mul (16|M16)             r56.0<1>:f    acc2.0<1;1,0>:f   r9.1<0;1,0>:f    {Compacted}
        send.ugm (16|M0)         null     r58     r54:2   0x2000000            0x640E0504           {A@2,$7} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.bti[2]
        send.ugm (16|M16)        null     r60     r56:2   0x2000000            0x640E0504           {A@1,$8} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.bti[2]
L1584:
        join (32|M0)                         L1600
L1600:
(W)     mov (8|M0)               r127.0<1>:f   r3.0<1;1,0>:f                    {Compacted}
(W)     send.gtwy (1|M0)         null     r127    null:0  0x0            0x02000010           {EOT,A@1} // wr:1+0, rd:0; end of thread
L1624:
        nop
(W)     mov (16|M0)              null<1>:ud    0x31DDF76D:ud
(W)     mov (16|M0)              null<1>:ud    0xB7C8D4B7:ud
(W)     mov (16|M0)              null<1>:ud    0x0:ud
(W)     mov (16|M0)              null<1>:ud    0x1:ud
        illegal
        illegal
        illegal
        illegal
        illegal
        illegal
        illegal
        illegal
        illegal
