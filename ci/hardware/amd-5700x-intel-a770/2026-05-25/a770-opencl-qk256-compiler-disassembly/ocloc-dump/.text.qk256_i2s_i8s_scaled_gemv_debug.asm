L0:
(W)     mov (8|M0)               r127.0<1>:ud  0x0:ud
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud
(W)     and (1|M0)               r127.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x60:ud              {I@2}
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
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x40:uw              {$2.src}
(W)     send.dc0 (8|M0)          r10      r127    null:0  0x0            0x021842FD           {A@1,$3} // wr:1h+0, rd:1; oword aligned block read x2
(W)     mov (8|M0)               r109.0<1>:ud  r0.0<1;1,0>:ud
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {A@1}
(W)     mul (1|M0)               acc0.0<1>:d   r7.3<0;1,0>:d     r109.2<0;1,0>:uw {A@1}
(W)     mach (1|M0)              r3.0<1>:d     r7.3<0;1,0>:d     r109.1<0;1,0>:d  {$0.dst}
(W)     mov (1|M0)               r4.0<1>:hf    0x1:hf
        add3 (16|M16)            r112.0<1>:d   r3.0<0;0>:d       r2.0<1;0>:uw      r7.0<0>:d        {I@1}
        add3 (16|M0)             r110.0<1>:d   r3.0<0;0>:d       r1.0<1;0>:uw      r7.0<0>:d
        sync.nop                             null                             {Compacted,$2.dst}
        cmp (16|M16)  (lt)f0.0   null<1>:d     r112.0<1;1,0>:ud  r9.4<0;1,0>:ud   {I@2}
        cmp (16|M16)  (lt)f1.0   null<1>:d     r112.0<1;1,0>:ud  r8.6<0;1,0>:ud
        cmp (16|M0)   (lt)f1.0   null<1>:d     r110.0<1;1,0>:ud  r9.4<0;1,0>:ud   {I@3}
(f0.0)  sel (16|M16)             acc0.0<1>:uw  r4.0<0;1,0>:uw    0x0:uw              {F@1}
(f1.0)  sel (16|M16)             acc2.0<1>:uw  r4.0<0;1,0>:uw    0x0:uw
(f1.0)  cmp (16|M0)   (lt)f1.0   null<1>:d     r110.0<1;1,0>:ud  r8.6<0;1,0>:ud
        and (16|M16)  (ne)f1.0   null<1>:uw    acc2.0<1;1,0>:uw  acc0.0<1;1,0>:uw
(~f1.0) goto (32|M0)                         L4720                  L4720
L536:
(W)     mov (8|M0)               r115.0<1>:w   0x76543210:v
(W)     mov (8|M0)               r114.0<1>:w   0x76543210:v
(W)     add (8|M0)               r115.8<1>:w   r115.0<1;1,0>:w   8:w               {I@2}
(W)     add (8|M0)               r114.8<1>:w   r114.0<1;1,0>:w   8:w               {I@2}
(W)     add (16|M0)              r115.0<1>:w   r115.0<1;1,0>:w   16:w               {I@2}
(W)     and (1|M0)               r3.0<1>:ud    sr0.0<0;1,0>:ud   16383:w               {A@1}
(W)     and (1|M0)               r4.0<1>:ud    r3.0<0;1,0>:ud    63:w               {A@1}
(W)     asr (1|M0)               r5.0<1>:ud    r3.0<0;1,0>:ud    1:w               {$1.dst}
(W)     mov (1|M0)               r6.0<1>:d     -64:w
(W)     mov (1|M0)               r9.5<1>:d     -8:w
(W)     bfn.0xF8 (1|M0)          r3.0<1>:ud    r5.0<0;0>:ud      r6.0<0;0>:ud      r4.0<0>:ud       {I@2} // s0&s1|s2
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r8.7<0;1,0>:d     0:w
(W)     cmp (16|M16)  (eq)f1.0   null<1>:d     r8.7<0;1,0>:d     0:w
(W)     and (1|M0)               r7.6<1>:ud    r3.0<0;1,0>:ud    7:w               {I@3}
(W)     asr (1|M0)               r8.0<1>:ud    r3.0<0;1,0>:ud    1:w
(W)     bfn.0xF8 (1|M0)          r3.0<1>:ud    r8.0<0;0>:ud      r9.5<0;0>:ud      r7.6<0>:ud       {I@1} // s0&s1|s2
(W)     mov (1|M0)               r116.0<1>:f   r3.0<0;1,0>:f                    {Compacted,I@1}
(~f1.0) goto (32|M0)                         L864                  L864
L816:
        mov (16|M0)              r117.0<1>:d   0:w
        mov (16|M16)             r119.0<1>:d   0:w
        goto (32|M0)                         L864                  L1456
L864:
        join (32|M0)                         L1456
L880:
(W)     mul (8|M0)               acc0.0<1>:d   r110.0<1;1,0>:d   r9.0<0;1,0>:uw   {Compacted}
        mach (8|M0)              r121.0<1>:d   r110.0<1;1,0>:d   r9.0<0;1,0>:d    {Compacted}
(W)     mul (8|M8)               acc0.0<1>:d   r111.0<1;1,0>:d   r9.0<0;1,0>:uw
        mach (8|M8)              r122.0<1>:d   r111.0<1;1,0>:d   r9.0<0;1,0>:d    {Compacted}
(W)     mul (8|M16)              acc0.0<1>:d   r112.0<1;1,0>:d   r9.0<0;1,0>:uw
        mov (16|M0)              r117.0<1>:d   0:w
        mov (16|M16)             r119.0<1>:d   0:w
        mach (8|M16)             r123.0<1>:d   r112.0<1;1,0>:d   r9.0<0;1,0>:d    {Compacted}
(W)     mul (8|M24)              acc0.0<1>:d   r113.0<1;1,0>:d   r9.0<0;1,0>:uw
(W)     mov (1|M0)               r116.1<1>:d   0:w
        mach (8|M24)             r124.0<1>:d   r113.0<1;1,0>:d   r9.0<0;1,0>:d    {Compacted}
L1016:
(W)     shr (1|M0)               r3.0<1>:ud    r116.1<0;1,0>:ud  2:w               {A@1}
(W)     and (1|M0)               r15.0<1>:d    r116.1<0;1,0>:d   31:w               {Compacted}
(W)     and (1|M0)               r4.0<1>:d     r3.0<0;1,0>:d     1073741760:d               {I@2}
(W)     and (1|M0)               r5.0<1>:d     r3.0<0;1,0>:d     32:w               {Compacted}
(W)     and (1|M0)               r20.0<1>:d    r116.1<0;1,0>:d   255:w               {Compacted}
        add3 (16|M0)             r11.0<1>:d    r121.0<1;0>:d     r4.0<0;0>:d       r5.0<0>:d        {I@2}
        add3 (16|M16)            r13.0<1>:d    r123.0<1;0>:d     r4.0<0;0>:d       r5.0<0>:d
        sync.nop                             null                             {Compacted,$3.dst}
        add3 (16|M0)             r16.0<1>:d    r11.0<1;0>:d      r15.0<0;0>:d      r10.1<0>:d       {I@2}
        add3 (16|M16)            r18.0<1>:d    r13.0<1;0>:d      r15.0<0;0>:d      r10.1<0>:d       {I@2}
        send.ugm (16|M0)         r22      r16     null:0  0x1000000            0x64280900           {A@2,$4} // wr:2+0, rd:2; load.ugm.d8u32.a32.ca.ca.bti[1]
        send.ugm (16|M16)        r24      r18     null:0  0x1000000            0x64280900           {A@1,$5} // wr:2+0, rd:2; load.ugm.d8u32.a32.ca.ca.bti[1]
(W)     and (1|M0)               r21.0<1>:d    r116.1<0;1,0>:d   128:w               {Compacted}
(W)     add (1|M0)               r27.0<1>:d    r116.1<0;1,0>:d   r10.0<0;1,0>:d   {Compacted}
(W)     add (1|M0)               r26.0<1>:d    r20.0<0;1,0>:d    -r21.0<0;1,0>:d  {Compacted,I@2}
(W)     send.ugm (1|M0)          r29      r27     null:0  0x0            0x62180900           {A@2,$6} // wr:1+0, rd:1; load.ugm.d8u32.a32.ca.ca.bti[0]
(W)     shr (1|M0)               r28.0<1>:ud   r26.0<0;1,0>:ud   4:w               {I@1}
(W)     add (1|M0)               r116.1<1>:d   r116.1<0;1,0>:d   1:w               {Compacted}
(W)     and (1|M0)               r30.0<1>:d    r28.0<0;1,0>:d    30:w               {Compacted,I@2}
        mov (16|M0)              r33.0<1>:d    r22.0<4;1,0>:ub                  {$4.dst}
(W)     add (1|M0)               r31.0<1>:d    -r30.0<0;1,0>:d   6:w               {Compacted,I@2}
        mov (16|M16)             r35.0<1>:d    r24.0<4;1,0>:ub                  {$5.dst}
(W)     and (1|M0)               r32.0<1>:d    r31.0<0;1,0>:d    30:w               {Compacted,I@2}
(W)     cmp (16|M0)   (lt)f0.0   null<1>:d     r116.1<0;1,0>:ud  r8.7<0;1,0>:ud
        shr (16|M0)              r33.0<1>:ud   r33.0<1;1,0>:ud   r32.0<0;1,0>:d   {I@2}
        shr (16|M16)             r35.0<1>:ud   r35.0<1;1,0>:ud   r32.0<0;1,0>:d
(W)     cmp (16|M16)  (lt)f0.0   null<1>:d     r116.1<0;1,0>:ud  r8.7<0;1,0>:ud
(W)     mov (1|M0)               r37.0<1>:w    r29.0<0;1,0>:b                   {$6.dst}
        and (16|M0)              r33.0<1>:d    r33.0<1;1,0>:d    3:w               {Compacted,I@4}
        and (16|M16)             r35.0<1>:d    r35.0<1;1,0>:d    3:w               {Compacted,I@4}
        mul (16|M0)              r33.0<1>:d    r33.0<1;1,0>:d    r37.0<0;1,0>:w   {I@2}
        mul (16|M16)             r35.0<1>:d    r35.0<1;1,0>:d    r37.0<0;1,0>:w   {I@2}
        add (16|M0)              r117.0<1>:d   r117.0<1;1,0>:d   r33.0<1;1,0>:d   {Compacted,I@2}
        add (16|M16)             r119.0<1>:d   r119.0<1;1,0>:d   r35.0<1;1,0>:d   {Compacted,I@2}
(f0.0)  goto.b (32|M0)                       L1456                  L1016
L1456:
        join (32|M0)                         L4720
L1472:
        shl (16|M0)              r3.0<1>:d     r114.0<1;1,0>:uw  2:w
(W)     mul (1|M0)               r5.0<1>:d     r116.0<0;1,0>:d   384:w               {Compacted}
        shl (16|M16)             r13.0<1>:d    r115.0<1;1,0>:uw  2:w
        add (16|M0)              r11.0<1>:d    r5.0<0;1,0>:d     r3.0<1;1,0>:d    {Compacted,I@2}
        addc (8|M0)              r15.0<1>:ud   r9.6<0;1,0>:ud    r11.0<1;1,0>:ud  {AccWrEn,Compacted,I@1}
        add (16|M0)              r17.0<1>:d    r114.0<1;1,0>:uw  32:w
        add (16|M16)             r19.0<1>:d    r5.0<0;1,0>:d     r13.0<1;1,0>:d   {Compacted}
        mov (8|M0)               r21.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M8)              r16.0<1>:ud   r9.6<0;1,0>:ud    r12.0<1;1,0>:ud  {AccWrEn}
        shl (16|M0)              r17.0<1>:d    r17.0<1;1,0>:d    2:w               {Compacted,I@4}
        mov (8|M8)               r22.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M16)             r23.0<1>:ud   r9.6<0;1,0>:ud    r19.0<1;1,0>:ud  {AccWrEn,I@5}
        add (16|M16)             r25.0<1>:d    r115.0<1;1,0>:uw  32:w
        add (16|M0)              r17.0<1>:d    r5.0<0;1,0>:d     r17.0<1;1,0>:d   {Compacted,I@4}
        mov (8|M16)              r27.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M24)             r24.0<1>:ud   r9.6<0;1,0>:ud    r20.0<1;1,0>:ud  {AccWrEn}
        shl (16|M16)             r25.0<1>:d    r25.0<1;1,0>:d    2:w               {Compacted,I@4}
        mov (8|M24)              r28.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M0)              r29.0<1>:ud   r9.6<0;1,0>:ud    r17.0<1;1,0>:ud  {AccWrEn,Compacted,I@5}
        add (16|M16)             r25.0<1>:d    r5.0<0;1,0>:d     r25.0<1;1,0>:d   {Compacted,I@3}
        mov (8|M0)               r31.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M8)              r30.0<1>:ud   r9.6<0;1,0>:ud    r18.0<1;1,0>:ud  {AccWrEn}
(W)     mov (1|M0)               r32.0<1>:hf   0x100:hf
        mov (8|M8)               r33.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M16)             r34.0<1>:ud   r9.6<0;1,0>:ud    r25.0<1;1,0>:ud  {AccWrEn,I@4}
        add3 (16|M0)             r36.0<1>:d    r5.0<0;0>:d       r32.0<0;0>:w      r3.0<1>:d        {F@1}
        mov (8|M16)              r38.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M24)             r35.0<1>:ud   r9.6<0;1,0>:ud    r26.0<1;1,0>:ud  {AccWrEn}
        add3 (16|M16)            r42.0<1>:d    r5.0<0;0>:d       r32.0<0;0>:w      r13.0<1>:d
        mov (8|M24)              r39.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M0)              r40.0<1>:ud   r9.6<0;1,0>:ud    r36.0<1;1,0>:ud  {AccWrEn,Compacted,I@5}
        add (16|M0)              r49.0<1>:d    r117.0<1;1,0>:d   -r9.1<0;1,0>:d   {Compacted}
        mov (8|M0)               r44.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M8)              r41.0<1>:ud   r9.6<0;1,0>:ud    r37.0<1;1,0>:ud  {AccWrEn}
        add (16|M16)             r51.0<1>:d    r119.0<1;1,0>:d   -r9.1<0;1,0>:d   {Compacted}
        mov (8|M8)               r45.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M16)             r46.0<1>:ud   r9.6<0;1,0>:ud    r42.0<1;1,0>:ud  {AccWrEn,I@7}
        mov (16|M0)              r74.0<1>:f    r9.2<0;1,0>:f                    {Compacted}
        mov (8|M16)              r48.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        addc (8|M24)             r47.0<1>:ud   r9.6<0;1,0>:ud    r43.0<1;1,0>:ud  {AccWrEn}
        mov (16|M16)             r76.0<1>:f    r9.2<0;1,0>:f                    {Compacted}
        mov (8|M24)              r53.0<1>:ud   acc0.0<1;1,0>:ud                 {Compacted}
        mov (16|M0)              r86.0<1>:f    r9.3<0;1,0>:f                    {Compacted}
        mov (16|M16)             r88.0<1>:f    r9.3<0;1,0>:f                    {Compacted}
        add (8|M0)               r58.1<2>:ud   r21.0<1;1,0>:ud   r9.7<0;1,0>:ud
        add (8|M8)               r60.1<2>:ud   r22.0<1;1,0>:ud   r9.7<0;1,0>:ud
        add (8|M16)              r62.1<2>:ud   r27.0<1;1,0>:ud   r9.7<0;1,0>:ud
        add (8|M24)              r64.1<2>:ud   r28.0<1;1,0>:ud   r9.7<0;1,0>:ud
        add (8|M0)               r66.1<2>:ud   r31.0<1;1,0>:ud   r9.7<0;1,0>:ud
        add (8|M8)               r68.1<2>:ud   r33.0<1;1,0>:ud   r9.7<0;1,0>:ud
        add (8|M16)              r70.1<2>:ud   r38.0<1;1,0>:ud   r9.7<0;1,0>:ud
        add (8|M24)              r72.1<2>:ud   r39.0<1;1,0>:ud   r9.7<0;1,0>:ud
        mov (16|M0)              r54.0<1>:f    r49.0<1;1,0>:d                   {Compacted}
        add (8|M0)               r78.1<2>:ud   r44.0<1;1,0>:ud   r9.7<0;1,0>:ud
        mov (16|M16)             r56.0<1>:f    r51.0<1;1,0>:d                   {Compacted}
        add (8|M8)               r80.1<2>:ud   r45.0<1;1,0>:ud   r9.7<0;1,0>:ud
        add (8|M16)              r82.1<2>:ud   r48.0<1;1,0>:ud   r9.7<0;1,0>:ud
        add (8|M24)              r84.1<2>:ud   r53.0<1;1,0>:ud   r9.7<0;1,0>:ud
        mov (8|M0)               r58.0<2>:d    r15.0<1;1,0>:d
        mov (8|M8)               r60.0<2>:d    r16.0<1;1,0>:d
        mov (8|M16)              r62.0<2>:d    r23.0<1;1,0>:d
        mov (8|M24)              r64.0<2>:d    r24.0<1;1,0>:d
        mov (8|M0)               r66.0<2>:d    r29.0<1;1,0>:d
        mov (8|M8)               r68.0<2>:d    r30.0<1;1,0>:d
        mov (8|M16)              r70.0<2>:d    r34.0<1;1,0>:d
        mov (8|M24)              r72.0<2>:d    r35.0<1;1,0>:d
        mov (8|M0)               r78.0<2>:d    r40.0<1;1,0>:d
        mov (8|M8)               r80.0<2>:d    r41.0<1;1,0>:d
        mov (8|M16)              r82.0<2>:d    r46.0<1;1,0>:d
        mov (8|M24)              r84.0<2>:d    r47.0<1;1,0>:d
        send.ugm (16|M0)         null     r58     r54:2   0x0            0x080E0584           {A@2,$7} // wr:4+2, rd:0; store.ugm.d32.a64.wb.wb
        send.ugm (16|M16)        null     r62     r56:2   0x0            0x080E0584           {A@1,$8} // wr:4+2, rd:0; store.ugm.d32.a64.wb.wb
        send.ugm (16|M0)         null     r66     r74:2   0x0            0x080E0584           {A@7,$9} // wr:4+2, rd:0; store.ugm.d32.a64.wb.wb
        send.ugm (16|M16)        null     r70     r76:2   0x0            0x080E0584           {A@5,$10} // wr:4+2, rd:0; store.ugm.d32.a64.wb.wb
        send.ugm (16|M0)         null     r78     r86:2   0x0            0x080E0584           {A@3,$11} // wr:4+2, rd:0; store.ugm.d32.a64.wb.wb
        send.ugm (16|M16)        null     r82     r88:2   0x0            0x080E0584           {A@1,$12} // wr:4+2, rd:0; store.ugm.d32.a64.wb.wb
(W)     and (1|M0)    (eq)f1.0   r96.0<1>:d    r9.2<0;1,0>:d     2139095040:d
(W)     mov (1|M0)               r100.0<1>:f   0x4F800000:f                               {Compacted}
(W)     mov (1|M0)               r104.0<1>:hf  0x1:hf
(W)     and (1|M0)               r103.0<1>:d   r9.2<0;1,0>:d     8388607:d
(W)     mov (1|M0)               r97.0<1>:uw   f1.0<0;1,0>:uw                   {Compacted}
(W)     and (1|M0)    (eq)f1.0   r101.0<1>:d   r9.2<0;1,0>:d     2139095040:d
        mul (16|M0)              r47.0<1>:f    r54.0<1;1,0>:f    r9.3<0;1,0>:f    {Compacted}
(W)     mov (1|M0)               r102.0<1>:uw  f1.0<0;1,0>:uw                   {Compacted}
(W)     mov (1|M0)               f1.0<1>:uw    r97.0<0;1,0>:uw                  {I@3}
(W&f1.0) sel (1|M0)              r105.0<1>:f   r100.0<0;1,0>:f   1.0:f               {F@3}
(W)     cmp (1|M0)    (ge)f1.0   null<1>:ud    r96.0<0;1,0>:ud   0x64000000:ud              {F@1}
(W)     mov (1|M0)               r106.0<1>:uw  f1.0<0;1,0>:uw                   {Compacted}
        send.ugm (16|M0)         r90      r66     null:0  0x0            0x08280580           {$13} // wr:4+0, rd:2; load.ugm.d32.a64.ca.ca
        send.ugm (16|M16)        r92      r70     null:0  0x0            0x08280580           {$14} // wr:4+0, rd:2; load.ugm.d32.a64.ca.ca
        send.ugm (16|M0)         r11      r58     null:0  0x0            0x08280580           {$15} // wr:4+0, rd:2; load.ugm.d32.a64.ca.ca
        send.ugm (16|M16)        r13      r62     null:0  0x0            0x08280580           {$0} // wr:4+0, rd:2; load.ugm.d32.a64.ca.ca
        send.ugm (16|M0)         r36      r78     null:0  0x0            0x08280580           {$1} // wr:4+0, rd:2; load.ugm.d32.a64.ca.ca
        send.ugm (16|M16)        r38      r82     null:0  0x0            0x08280580           {$2} // wr:4+0, rd:2; load.ugm.d32.a64.ca.ca
        and (16|M0)   (eq)f0.0   r94.0<1>:d    r90.0<1;1,0>:d    2139095040:d               {$13.dst}
        and (16|M16)  (eq)f0.0   r98.0<1>:d    r92.0<1;1,0>:d    2139095040:d               {$14.dst}
(f0.0)  sel (16|M0)              r107.0<1>:f   r100.0<0;1,0>:f   1.0:f               {Compacted}
        cmp (16|M0)   (eq)f0.0   null<1>:d     r94.0<1;1,0>:d    0:w               {Compacted,A@1}
(f0.0)  sel (16|M16)             r3.0<1>:f     r100.0<0;1,0>:f   1.0:f
(W)     mov (1|M0)               r5.0<1>:ud    f0.0<0;1,0>:ud                   {Compacted}
(W)     and (1|M0)    (eq)f0.0   r7.6<1>:d     r9.2<0;1,0>:d     2139095040:d               {F@1}
        cmp (16|M16)  (ge)f0.0   null<1>:d     r98.0<1;1,0>:ud   0x64000000:ud              {I@4}
(W)     mov (1|M0)               f1.0<1>:ud    r5.0<0;1,0>:ud                   {I@3}
(W)     mov (1|M0)               r8.0<1>:uw    f0.0<0;1,0>:uw                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:uw    r106.0<0;1,0>:uw
(f0.0)  mov (16|M16)             r3.0<1>:f     0x2F800000:f
(f1.0)  sel (16|M0)              r6.0<1>:uw    r104.0<0;1,0>:uw  0x0:uw              {Compacted}
(W&~f0.0) sel (1|M0)             r10.2<1>:f    r105.0<0;1,0>:f   0x2F800000:f               {$3.dst}
        cmp (16|M0)   (ge)f0.0   null<1>:d     r94.0<1;1,0>:ud   0x64000000:ud              {F@1}
(W)     cmp (1|M0)    (ge)f1.0   null<1>:ud    r101.0<0;1,0>:ud  0x64000000:ud
(W)     cmp (16|M16)  (eq)f1.0   null<1>:d     r103.0<0;1,0>:d   0:w
(f0.0)  mov (16|M0)              r107.0<1>:f   0x2F800000:f                               {Compacted}
(W)     mov (1|M0)               f0.0<1>:uw    r8.0<0;1,0>:uw                   {A@1}
(W)     mov (1|M0)               r9.10<1>:uw   f1.0<0;1,0>:uw                   {Compacted}
(W)     mov (1|M0)               f1.0<1>:uw    r102.0<0;1,0>:uw
(W&~f1.0) cmp (16|M16) (eq)f1.0  null<1>:d     r96.0<0;1,0>:d    0:w
(W&f0.0) sel (1|M0)              r17.0<1>:f    r100.0<0;1,0>:f   1.0:f
        and (16|M0)   (eq)f0.0   null<1>:d     r90.0<1;1,0>:d    8388607:d               {F@1}
(W&f1.0) sel (1|M0)              r15.0<1>:f    r100.0<0;1,0>:f   1.0:f
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r103.0<0;1,0>:d   0:w               {F@1}
(W)     mov (1|M0)               r18.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
(W)     not (1|M16)              r33.1<1>:uw   f1.1<0;1,0>:uw
(W)     mov (1|M0)               f0.0<1>:ud    r18.0<0;1,0>:ud                  {I@2}
        and (16|M16)  (eq)f0.0   null<1>:d     r92.0<1;1,0>:d    8388607:d
(W&~f1.0) cmp (16|M0) (eq)f1.0   null<1>:d     r96.0<0;1,0>:d    0:w
(W)     mov (1|M0)               r18.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
(W)     cmp (1|M0)    (ge)f0.0   null<1>:ud    r7.6<0;1,0>:ud    0x64000000:ud
(W)     not (1|M0)               r33.0<1>:uw   f1.0<0;1,0>:uw                   {Compacted}
(W)     mul (1|M0)               r16.0<1>:f    r9.2<0;1,0>:f     r10.2<0;1,0>:f
        mul (16|M16)             r27.0<1>:f    r92.0<1;1,0>:f    r3.0<1;1,0>:f    {Compacted}
(W)     mov (1|M0)               r19.0<1>:uw   f0.0<0;1,0>:uw                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    r18.0<0;1,0>:ud                  {I@4}
        sync.nop                             null                             {Compacted,F@2}
(W)     math.inv (1|M0)          r30.0<1>:f    r16.0<0;1,0>:f                   {$4}
(f0.0)  sel (16|M0)              r20.0<1>:uw   r104.0<0;1,0>:uw  0x0:uw              {Compacted}
(f0.0)  sel (16|M16)             r21.0<1>:uw   r104.0<0;1,0>:uw  0x0:uw
(W)     mov (1|M0)               f0.0<1>:ud    r5.0<0;1,0>:ud
        cmp (16|M16)  (eq)f0.0   null<1>:d     r98.0<1;1,0>:d    0:w
        mul (16|M0)              acc2.0<1>:f   r30.0<0;1,0>:f    r54.0<1;1,0>:f   {Compacted,$4.dst}
(W)     mov (1|M0)               r5.0<1>:ud    f0.0<0;1,0>:ud                   {Compacted}
(W)     and (1|M0)    (eq)f0.0   null<1>:d     r9.2<0;1,0>:d     8388607:d
(f0.0)  sel (16|M16)             r22.0<1>:uw   r104.0<0;1,0>:uw  0x0:uw
        mul (16|M16)             acc0.0<1>:f   r30.0<0;1,0>:f    r56.0<1;1,0>:f   {Compacted}
(W)     mov (1|M0)               r23.0<1>:uw   f0.0<0;1,0>:uw                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:uw    r9.10<0;1,0>:uw
        mul (16|M0)              r60.0<1>:f    acc2.0<1;1,0>:f   r10.2<0;1,0>:f   {Compacted,$15.src}
        mul (16|M16)             r62.0<1>:f    acc0.0<1;1,0>:f   r10.2<0;1,0>:f   {Compacted,$0.src}
(W&~f0.0) sel (1|M0)             r24.0<1>:f    r15.0<0;1,0>:f    0x2F800000:f
(W)     cmp (16|M0)   (eq)f0.0   null<1>:d     r103.0<0;1,0>:d   0:w               {Compacted,F@1}
(W)     mul (1|M0)               r32.0<1>:f    r9.2<0;1,0>:f     r24.0<0;1,0>:f   {Compacted}
        mul (16|M0)              r25.0<1>:f    r90.0<1;1,0>:f    r107.0<1;1,0>:f  {Compacted}
(W)     mov (1|M0)               r29.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
(W)     math.inv (1|M0)          r35.0<1>:f    r32.0<0;1,0>:f                   {@2,$5}
(W)     mov (1|M0)               f0.0<1>:ud    r29.0<0;1,0>:ud                  {I@1}
(W)     cmp (16|M16)  (eq)f0.0   null<1>:d     r103.0<0;1,0>:d   0:w
        math.inv (16|M16)        r27.0<1>:f    r27.0<1;1,0>:f                   {$6}
(W)     mov (1|M0)               r29.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:uw    r19.0<0;1,0>:uw
(W)     mov (1|M0)               f1.0<1>:ud    r29.0<0;1,0>:ud                  {I@2}
        sync.nop                             null                             {Compacted,F@1}
        math.inv (16|M0)         r25.0<1>:f    r25.0<1;1,0>:f                   {$13}
(W&~f1.0) cmp (16|M0) (eq)f1.0   null<1>:d     r7.6<0;1,0>:d     0:w
(W&~f0.0) sel (1|M0)             r31.0<1>:f    r17.0<0;1,0>:f    0x2F800000:f
(W)     mov (1|M0)               f0.0<1>:uw    r23.0<0;1,0>:uw                  {F@1}
(W)     mov (1|M0)               r29.0<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f1.0<1>:ud    r5.0<0;1,0>:ud
        or (16|M0)    (ne)f1.0   null<1>:uw    r6.0<1;1,0>:uw    r20.0<1;1,0>:uw
(W&~f0.0) cmp (1|M0)  (eq)f0.0   null<1>:d     r101.0<0;1,0>:d   0:w
(W)     mul (1|M0)               r64.0<1>:f    r35.0<0;1,0>:f    r9.3<0;1,0>:f    {Compacted,$5.dst}
(W)     mov (1|M0)               r5.0<1>:ud    f1.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f1.0<1>:ud    r33.0<0;1,0>:ud
(W)     mov (1|M0)               r23.0<1>:uw   f0.0<0;1,0>:uw                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    r5.0<0;1,0>:ud                   {I@3}
        or (16|M16)   (ne)f0.0   null<1>:uw    r22.0<1;1,0>:uw   r21.0<1;1,0>:uw
(f1.0)  cmp (16|M0)   (eq)f1.0   null<1>:f     r54.0<1;1,0>:f    r9.2<0;1,0>:f    {I@4}
(W)     mov (1|M0)               r5.0<1>:ud    f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    r29.0<0;1,0>:ud
(W)     mov (1|M0)               r33.0<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W)     not (1|M0)               r44.0<1>:uw   r23.0<0;1,0>:uw                  {I@6}
(W)     not (1|M0)               r45.0<1>:uw   r5.0<0;1,0>:uw                   {I@4}
(W&~f0.0) cmp (16|M16) (eq)f0.0  null<1>:d     r7.6<0;1,0>:d     0:w
(W)     mov (1|M0)               f1.0<1>:uw    r44.0<0;1,0>:uw                  {A@1}
(W)     not (1|M16)              r45.1<1>:uw   r5.1<0;1,0>:uw
(W)     mov (1|M0)               r29.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    r33.0<0;1,0>:ud
(W&f1.0) cmp (1|M0)   (eq)f1.0   null<1>:f     r9.3<0;1,0>:f     r9.2<0;1,0>:f    {I@4}
(W)     mul (1|M0)               r34.0<1>:f    r9.2<0;1,0>:f     r31.0<0;1,0>:f   {Compacted}
(W)     mul (1|M0)               r77.0<1>:f    r64.0<0;1,0>:f    r24.0<0;1,0>:f   {Compacted,$10.src}
(f0.0)  cmp (16|M16)  (eq)f0.0   null<1>:f     r56.0<1;1,0>:f    r9.2<0;1,0>:f    {I@1}
(W)     mov (1|M0)               r44.0<1>:uw   f1.0<0;1,0>:uw                   {Compacted}
(W)     math.inv (1|M0)          r46.0<1>:f    r34.0<0;1,0>:f                   {@3,$14}
(W)     mov (1|M0)               r33.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
(W)     not (1|M0)               f0.0<1>:uw    r29.0<0;1,0>:uw                  {F@1}
(W)     mov (1|M0)               f1.0<1>:ud    r33.0<0;1,0>:ud                  {I@2}
(W)     not (1|M16)              f0.1<1>:uw    r29.1<0;1,0>:uw
        mul (16|M16)             acc2.0<1>:f   r56.0<1;1,0>:f    r9.3<0;1,0>:f    {Compacted}
(~f1.0) sel (16|M0)              r73.0<1>:f    r60.0<1;1,0>:f    1.0:f               {$9.src}
(~f1.0) sel (16|M16)             r75.0<1>:f    r62.0<1;1,0>:f    1.0:f
(W)     mov (1|M0)               f1.0<1>:ud    r45.0<0;1,0>:ud                  {F@1}
        sync.nop                             null                             {Compacted,$6.dst}
        mul (16|M16)             acc0.0<1>:f   r27.0<1;1,0>:f    r13.0<1;1,0>:f   {Compacted,$0.dst}
        sync.nop                             null                             {Compacted,$13.dst}
        mul (16|M0)              r25.0<1>:f    r25.0<1;1,0>:f    r11.0<1;1,0>:f   {Compacted,$15.dst}
(f1.0)  cmp (16|M0)   (eq)f1.0   null<1>:f     r11.0<1;1,0>:f    r90.0<1;1,0>:f   {I@1}
        mul (16|M0)              r65.0<1>:f    r46.0<0;1,0>:f    r47.0<1;1,0>:f   {Compacted,$14.dst}
(W)     mov (1|M0)               r45.0<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(f0.0)  cmp (16|M0)   (eq)f0.0   null<1>:f     r47.0<1;1,0>:f    r9.2<0;1,0>:f
(W)     mov (1|M0)               f1.0<1>:ud    r45.0<0;1,0>:ud                  {A@1}
        mul (16|M16)             r67.0<1>:f    r46.0<0;1,0>:f    acc2.0<1;1,0>:f
(f0.0)  cmp (16|M16)  (eq)f0.0   null<1>:f     acc2.0<1;1,0>:f   r9.2<0;1,0>:f
(f1.0)  cmp (16|M16)  (eq)f1.0   null<1>:f     r13.0<1;1,0>:f    r92.0<1;1,0>:f   {I@1}
        mul (16|M16)             acc0.0<1>:f   acc0.0<1;1,0>:f   r3.0<1;1,0>:f    {Compacted}
(W)     mov (1|M0)               r45.0<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f1.0<1>:uw    r44.0<0;1,0>:uw                  {F@2}
        mul (16|M0)              acc2.0<1>:f   r25.0<1;1,0>:f    r107.0<1;1,0>:f  {Compacted}
        shl (16|M0)              r78.0<1>:d    r110.0<1;1,0>:d   5:w               {Compacted,$1.src}
(W&~f1.0) sel (1|M0)             r102.0<1>:f   r77.0<0;1,0>:f    1.0:f
(W)     mov (1|M0)               f1.0<1>:ud    r45.0<0;1,0>:ud                  {A@1}
        shl (16|M16)             r80.0<1>:d    r112.0<1;1,0>:d   5:w               {Compacted}
        mul (16|M0)              r82.0<1>:f    r73.0<1;1,0>:f    r9.3<0;1,0>:f    {Compacted,$2.src}
        mul (16|M16)             r90.0<1>:f    r75.0<1;1,0>:f    r9.3<0;1,0>:f    {Compacted}
        mul (16|M0)              r98.0<1>:f    r65.0<1;1,0>:f    r31.0<0;1,0>:f   {Compacted}
        mul (16|M16)             r100.0<1>:f   r67.0<1;1,0>:f    r31.0<0;1,0>:f   {Compacted}
(~f1.0) sel (16|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   1.0:f
(~f1.0) sel (16|M16)             acc0.0<1>:f   acc0.0<1;1,0>:f   1.0:f
        mov (16|M0)              r15.0<1>:f    r49.0<1;1,0>:f                   {Compacted}
        mov (16|M16)             r17.0<1>:f    r119.0<1;1,0>:f                  {Compacted}
        mov (16|M16)             r19.0<1>:d    r9.1<0;1,0>:d
        mov (16|M16)             r21.0<1>:f    r51.0<1;1,0>:f                   {Compacted}
        mov (16|M0)              r29.0<1>:f    r9.3<0;1,0>:f                    {Compacted}
        mov (16|M0)              r11.0<1>:f    r117.0<1;1,0>:f                  {Compacted}
        mov (16|M0)              r13.0<1>:d    r9.1<0;1,0>:d
        mul (16|M16)             r3.0<1>:d     r112.0<1;1,0>:d   12:w               {Compacted}
        mov (16|M0)              r27.0<1>:f    r9.2<0;1,0>:f                    {Compacted}
        mul (16|M0)              r107.0<1>:d   r110.0<1;1,0>:d   12:w               {Compacted}
        add (16|M0)              r23.0<1>:d    r78.0<1;1,0>:d    16:w               {Compacted,I@7}
        mul (16|M0)              r86.0<1>:f    r102.0<0;1,0>:f   r54.0<1;1,0>:f   {Compacted,$11.src}
        mul (16|M16)             r94.0<1>:f    r102.0<0;1,0>:f   r56.0<1;1,0>:f   {Compacted}
        add (16|M16)             r25.0<1>:d    r80.0<1;1,0>:d    16:w               {Compacted,I@6}
        mov (16|M0)              r33.0<1>:f    r82.0<1;1,0>:f                   {Compacted}
        mov (16|M16)             r41.0<1>:f    r90.0<1;1,0>:f                   {Compacted}
(~f0.0) sel (16|M0)              r84.0<1>:f    r98.0<1;1,0>:f    1.0:f
(~f0.0) sel (16|M16)             r92.0<1>:f    r100.0<1;1,0>:f   1.0:f
        mov (16|M0)              r31.0<1>:f    r54.0<1;1,0>:f                   {Compacted}
        sync.nop                             null                             {Compacted,$1.dst}
        mul (16|M0)              r88.0<1>:f    acc2.0<1;1,0>:f   r36.0<1;1,0>:f   {Compacted,$12.src}
        mul (16|M16)             r96.0<1>:f    acc0.0<1;1,0>:f   r38.0<1;1,0>:f   {Compacted,$2.dst}
        mov (16|M16)             r35.0<1>:f    r9.2<0;1,0>:f                    {Compacted}
        mov (16|M16)             r37.0<1>:f    r9.3<0;1,0>:f                    {Compacted}
        mov (16|M16)             r39.0<1>:f    r56.0<1;1,0>:f                   {Compacted}
        sync.nop                             null                             {Compacted,I@3}
        send.ugm (16|M0)         null     r107    r11:6   0x2000000            0x640E2504           {$4} // wr:2+6, rd:0; store.ugm.d32x3.a32.wb.wb.bti[2]
        send.ugm (16|M16)        null     r3      r17:6   0x2000000            0x640E2504           {$5} // wr:2+6, rd:0; store.ugm.d32x3.a32.wb.wb.bti[2]
        send.ugm (16|M0)         null     r78     r27:8   0x3000000            0x640E3504           {A@6,$6} // wr:2+8, rd:0; store.ugm.d32x4.a32.wb.wb.bti[3]
        send.ugm (16|M16)        null     r80     r35:8   0x3000000            0x640E3504           {A@1,$9} // wr:2+8, rd:0; store.ugm.d32x4.a32.wb.wb.bti[3]
        sync.nop                             null                             {Compacted,I@2}
        send.ugm (16|M0)         null     r23     r82:8   0x3000000            0x640E3504           {$10} // wr:2+8, rd:0; store.ugm.d32x4.a32.wb.wb.bti[3]
        sync.nop                             null                             {Compacted,I@1}
        send.ugm (16|M16)        null     r25     r90:8   0x3000000            0x640E3504           {$11} // wr:2+8, rd:0; store.ugm.d32x4.a32.wb.wb.bti[3]
L4720:
        join (32|M0)                         L4736
L4736:
(W)     mov (8|M0)               r127.0<1>:f   r109.0<1;1,0>:f                  {Compacted,$3.src}
(W)     send.gtwy (1|M0)         null     r127    null:0  0x0            0x02000010           {EOT,A@1} // wr:1+0, rd:0; end of thread
L4760:
        nop
(W)     mov (16|M0)              null<1>:ud    0x5CF3FBBB:ud
(W)     mov (16|M0)              null<1>:ud    0xA5DA614B:ud
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
