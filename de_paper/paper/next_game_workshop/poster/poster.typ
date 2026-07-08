// =============================================================================
//  A0 Portrait Academic Poster  —  841 mm x 1189 mm  (84.1 x 118.9 cm)
//  "Learned Coordination Conventions in Cooperative MARL"
//
//  Build:   typst compile poster.typ poster.pdf
//  Requires: Typst >= 0.11  (uses grid.cell colspan + table.hline)
//  Self-contained: no external images, no packages, no network assets.
//
//  Layout knobs (top of file): BODY, GUT, SH, R.
//  If a column overruns the page, drop BODY by 1pt — everything is relative.
// =============================================================================

#let BODY = 21pt      // base body size
#let GUT  = 18mm      // inter-column gutter
#let SH   = 5pt       // drop-shadow offset
#let R    = 10pt      // corner radius

// ------------------------------- palette ------------------------------------
#let navy       = rgb("#0B2545")   // primary
#let navy-soft  = rgb("#13315C")   // secondary banner
#let cyan       = rgb("#00A6D6")   // technical accent
#let cyan-pale  = rgb("#E3F4FB")   // accent tint
#let ink        = rgb("#1A1A1A")
#let muted      = rgb("#5A6472")
#let border-col = rgb("#C9D3DE")
#let shadow-col = rgb("#BFC8D2")
#let page-bg    = rgb("#F7F9FB")
#let ok-green   = rgb("#1B7F4C")
#let tie-amber  = rgb("#B4761A")
#let warn-rust  = rgb("#9A3412")

// -------------------------------- page --------------------------------------
#set page(
  width: 841mm,
  height: 1189mm,
  margin: (x: 22mm, top: 20mm, bottom: 16mm),
  fill: page-bg,
)

#set text(
  font: ("Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"),
  size: BODY,
  fill: ink,
  lang: "en",
)
#set par(justify: true, leading: 0.62em, spacing: 0.95em)
#show strong: set text(fill: navy)
#show raw: set text(font: "DejaVu Sans Mono", size: 0.88em)

// ============================== components ===================================

// Section card: coloured header banner + faux drop shadow.
// Typst has no box-shadow; `move` preserves the layout box, so the parent's
// fill peeks out along the bottom/right edges. `pad` reclaims the overhang.
#let card(title, body, accent: navy, tint: white) = pad(
  left: SH, top: SH,
  block(
    width: 100%, fill: shadow-col, radius: R,
    move(dx: -SH, dy: -SH,
      block(
        width: 100%, fill: tint, radius: R,
        stroke: 1.4pt + border-col, clip: true,
        stack(
          block(
            width: 100%, fill: accent, inset: (x: 18pt, y: 13pt),
            text(size: 31pt, weight: 800, fill: white, tracking: 0.6pt, upper(title)),
          ),
          block(width: 100%, inset: (x: 18pt, y: 16pt), body),
        ),
      ),
    ),
  ),
)

// Downward flow arrow: stem + triangular head.
#let arrowdown(h: 20pt, col: navy, dashed: false) = align(center,
  grid(
    columns: (auto,), align: center, gutter: 0pt,
    if dashed {
      box(width: 14pt, height: h, place(center, line(
        start: (0pt, 0pt), end: (0pt, h),
        stroke: (paint: col, thickness: 3pt, dash: "dashed"),
      )))
    } else {
      rect(width: 3.4pt, height: h, fill: col, stroke: none)
    },
    polygon(fill: col, (0pt, 0pt), (13pt, 0pt), (6.5pt, 11pt)),
  ),
)

// [PASS] / [TIED] pill
#let badge(txt, col) = box(
  fill: col, radius: 5pt, inset: (x: 9pt, y: 4pt), baseline: 3pt,
  text(fill: white, size: 16pt, weight: 700, tracking: 0.5pt, txt),
)

// Node in the architecture diagram
#let node(body, fill: white, edge: navy, dash: none, size: 18pt) = block(
  width: 100%, fill: fill, radius: 6pt,
  stroke: (paint: edge, thickness: 2.2pt, dash: dash),
  inset: (x: 6pt, y: 11pt),
  align(center, text(size: size, weight: 600, fill: navy, body)),
)

// Callout with a thick accent rail on the left
#let callout(body, rail: cyan, fill: white) = block(
  width: 100%, fill: fill, radius: 6pt,
  stroke: (left: 5pt + rail, rest: 1pt + border-col),
  inset: (x: 14pt, y: 12pt),
  body,
)

// Booktabs-style rules — thick enough to survive cloth/large-format printing
#let toprule    = table.hline(stroke: 2.4pt + navy)
#let midrule    = table.hline(stroke: 1.1pt + border-col)
#let bottomrule = table.hline(stroke: 2.4pt + navy)

// Horizontal bar row. `lw`/`nw` are the label/number column widths so the same
// helper works in a full-width card and in a half-width sub-panel.
#let winbar(label, v, col, note, lw: 200pt, nw: 150pt, ls: 19pt) = grid(
  columns: (lw, 1fr, nw), align: horizon, gutter: 12pt,
  text(size: ls, weight: 600, label),
  box(width: 100%, height: 28pt, fill: rgb("#E6EAEF"), radius: 4pt,
    place(left + horizon, box(width: v * 100%, height: 28pt, fill: col, radius: 4pt))),
  align(right, text(size: ls, weight: 700, fill: navy, note)),
)

// Routing-matrix heatmap: linear ramp over the observed range 0.35 .. 0.65
#let heat(v) = {
  let t = calc.clamp((v - 0.35) / 0.30, 0.0, 1.0)
  color.mix((cyan-pale, (1.0 - t) * 100%), (navy, t * 100%))
}
#let heatcell(v) = block(
  width: 100%, height: 74pt, fill: heat(v), radius: 4pt, inset: 6pt,
  align(center + horizon, text(
    size: 22pt, weight: 700,
    fill: if v > 0.50 { white } else { navy },
    [#v],
  )),
)
#let deadcell = block(
  width: 100%, height: 74pt, fill: rgb("#EDF1F4"), radius: 4pt,
  align(center + horizon, text(size: 26pt, fill: muted, weight: 700, [—])),
)

// Dashed placeholder frame (logo / QR)
#let logobox(label) = block(
  width: 100%, height: 108pt, radius: 8pt, fill: white,
  stroke: (paint: border-col, thickness: 2pt, dash: "dashed"),
  align(center + horizon, text(size: 17pt, fill: muted, weight: 600, label)),
)

// =============================================================================
//  HEADER
// =============================================================================
#block(
  width: 100%, radius: R, fill: navy, inset: (x: 34pt, y: 28pt),
  grid(
    columns: (190pt, 1fr, 190pt), gutter: 30pt, align: horizon,

    logobox[INSTITUTION\ LOGO],

    align(center)[
      #text(size: 20pt, fill: cyan, weight: 700, tracking: 3pt)[
        ICML 2026 WORKSHOP #h(6pt)·#h(6pt) COOPERATIVE MULTI-AGENT RL
      ]
      #v(14pt)
      #text(size: 58pt, weight: 800, fill: white, hyphenate: false)[
        Learned Coordination Conventions\ in Cooperative MARL
      ]
      #v(10pt)
      #text(size: 31pt, weight: 400, fill: rgb("#B9CBDD"), style: "italic")[
        Measuring the Translation Gap Between\
        Theory-Informed Roles and Learned Routing
      ]
      #v(20pt)
      #line(length: 55%, stroke: 1.6pt + cyan)
      #v(16pt)
      #text(size: 27pt, weight: 700, fill: white)[Yoosung Hong]
      #v(4pt)
      #text(size: 21pt, fill: rgb("#B9CBDD"))[
        Independent Researcher #h(14pt)·#h(14pt)
        `yoosunghong.main@gmail.com` #h(14pt)·#h(14pt) `github: yoosung.dev`
      ]
    ],

    logobox[QR CODE\ (paper + code)],
  ),
)

#v(16pt)

// =============================================================================
//  BODY — three balanced columns
// =============================================================================
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: GUT,

  // ===========================================================================
  //  COLUMN 1 — Introduction & Motivation
  // ===========================================================================
  [
    // ---------------------------------------------------------------- Abstract
    #card("1. Abstract", accent: navy-soft, tint: cyan-pale)[
      #text(size: 21.5pt)[
        Role-semantic assignments supply *priors* over how heterogeneous agents
        may coordinate; cooperative MARL agents instead settle on *conventions*
        through decentralized, non-stationary learning — with no guarantee the
        resulting structure matches those priors.

        We ask whether the convention selected by training is *legible from the
        policy architecture.* Our diagnostic combines a *role-routing matrix*, a
        *formation-sensitivity* score ($Delta_max$), and *gradient / occlusion
        attribution*, evaluated across MiniGrid and SMACv2 (Terran), five MAPPO
        conditions, and a 3v3–9v9 scaling study.

        Label-conditioned attention produces *4.5× more concentrated* role
        routing than flat MLPs (entry std $0.246$ vs. $0.055$). The signature
        survives scaling, transfers *zero-shot* 3v3→9v9 above the from-scratch
        baseline ($0.224$ vs. $0.110$), and is invariant to ally-slot padding.
        Under 5-seed re-evaluation *3 of 4* theory-informed predictions hold and
        one is tied — the diagnostic is itself seed-sensitive.
      ]
    ]

    #v(12pt)

    // ------------------------------------------------------------ Introduction
    #card("2. Introduction & Motivation")[
      The central question is no longer *what equilibrium should exist,* but
      *what convention a learning system actually settles on* — an equilibrium
      selection problem that theory alone cannot resolve.

      #v(10pt)
      #callout[
        *The translation gap.* Classical role reasoning gives strategic priors
        under explicit rationality assumptions. MARL agents select a convention
        via decentralized gradient updates in non-stationary environments. There
        is no a priori reason the two should align.
      ]
      #v(12pt)

      *Where existing cooperative MARL falls short:*
      #v(4pt)
      #set list(marker: text(fill: cyan, size: 22pt)[▸], spacing: 0.75em)
      - *QMIX / MAPPO expose no structural path for role-pair routing.* Credit is
        assigned to _agents_, never to _role-to-role interactions_ — the
        coordination structure dissolves into weight products.
      - *Flat MLPs make "which entity do I condition on, given my role?"
        implicit.* Concatenating a role one-hot does not create a readable
        routing channel: C3 vs. C5 differ by only $+1.6$ pp, within noise.
      - *Interpretability is retrofitted, not architectural.* Post-hoc saliency
        over a monolithic policy cannot separate spatial from role-semantic
        contributions.
      - *Structure outlives performance.* At 6v6/9v9 the win-rate gaps collapse
        into seed noise, yet the routing signature persists — performance alone
        is a lossy probe of coordination.

      #v(12pt)
      *Why role-conditioned routing is the right unit of analysis.* Placing the
      role one-hot in *both* `self_tok` (→ Query) and every `ally_tok` slot
      (→ Key/Value) makes the attention score a *direct function of the role
      pair* $(r_i, r_j)$ — turning the learned convention into something you can
      *read off a matrix* rather than infer from behaviour.
    ]

    #v(12pt)

    // ------------------------------------------------- Setup: domains + conds
    #card("Setup — Domains & Five Conditions", accent: navy-soft)[
      #set par(justify: false)

      #text(size: 19.5pt)[
        *Domain 1 — MiniGrid.* Custom $15 times 15$ zone-capture task; three
        role-specialized agents under role-specific reward shaping.
        #linebreak()
        *Domain 2 — SMACv2 (Terran).* Marine = *Strike*, Marauder = *Vanguard*,
        Medivac = *Support*. SMACv2 exposes no role labels, so we inject a 3-dim
        unit-type one-hot into the agent's own token *and* every ally token —
        preserving the structural condition the routing mechanism requires.
      ]

      #v(12pt)
      #set text(size: 17pt)
      #table(
        columns: (auto, 1fr, auto, auto),
        align: (left, left, center, center),
        stroke: none,
        inset: (x: 4pt, y: 6.5pt),
        fill: (_, y) => if calc.odd(y) { rgb("#F4F7FA") } else { none },
        toprule,
        text(weight: 700)[Cond.],
        text(weight: 700)[Architecture],
        text(weight: 700)[Labels],
        text(weight: 700)[Policy],
        midrule,
        strong[C1 Full],      [Intra-Set + Cross-Attn], [✓], [per-role],
        strong[C2],           [Cross-Attn only],        [✓], [per-role],
        strong[C3 Role+MLP],  [Flat MLP],               [✓], [per-role],
        strong[C4 No-Role],   [Intra-Set + Cross-Attn], [✗], [*shared*],
        strong[C5 Flat MLP],  [Flat MLP],               [✗], [per-role],
        bottomrule,
      )

      #v(12pt)
      #set text(size: BODY)
      #callout(rail: tie-amber, fill: rgb("#FFF6E8"))[
        #text(size: 18pt)[
          *Read the ablations carefully.* C1 vs. C4 bundles label removal with a
          per-role → shared-policy change — it *upper-bounds* the label effect
          rather than isolating it. C3 vs. C5 is the clean label-in-input test
          for MLPs: $+1.6$ pp, within noise. C4 vs. C5 additionally confounds a
          $3 times$ data-per-parameter advantage from sharing.
        ]
      ]
    ]

    #v(12pt)

    // ------------------------------------------------------------ Predictions
    #card("Theory-Informed Predictions", accent: navy-soft)[
      #set par(justify: false, leading: 0.55em)
      #text(size: 19.5pt, fill: muted)[
        Derived from designer role semantics, then tested over *5 seeds.*
      ]
      #v(12pt)

      #stack(
        spacing: 13pt,

        [#badge("PASS", ok-green) #h(6pt) *P1 — Concentration.* Routing entry std
         $≥ 2×$ the MLP gradient-attribution std.
         #linebreak()
         #text(size: 18pt, fill: muted)[Observed *4.5×* — $0.246$ vs. $0.055$.]],

        [#badge("PASS", ok-green) #h(6pt) *P2 — Label necessity.* Zeroing role
         labels under shared attention costs $> 5$ pp.
         #linebreak()
         #text(size: 18pt, fill: muted)[Observed *−14.0 pp.* Bundles label removal
         with policy sharing → an *upper bound*, not an isolated effect.]],

        [#badge("TIED", tie-amber) #h(6pt) *P3 — Formation sensitivity.*
         $Delta_max$ strictly highest for Vanguard.
         #linebreak()
         #text(size: 18pt, fill: muted)[Support $0.074$, Vanguard $0.066$ — gap
         smaller than either seed std. The qualitative intent (*non-DPS roles are
         formation-sensitive*; Strike $0.024$ is not) is recovered.]],

        [#badge("PASS", ok-green) #h(6pt) *P4 — Strike → Support.* Strike routes
         preferentially to Support.
         #linebreak()
         #text(size: 18pt, fill: muted)[Holds in *4/5 seeds* at both 3v3 and 9v9.
         The earlier 3-seed Strike→Vanguard result was a *small-$n$ artifact.*]],
      )
    ]
  ],

  // ===========================================================================
  //  COLUMN 2 — Methodology (the core)
  // ===========================================================================
  [
    // ----------------------------------------------------------- Architecture
    #card("3. Architecture (Cond. 1 — Full)")[
      #set par(justify: false)
      #let TOK = rgb("#EEF2F6")

      #grid(
        columns: (1fr, 1fr, 1fr, 1fr),
        gutter: 9pt,
        align: horizon + center,

        // ---- layer 0 : entity tokens  (role one-hot lives in the cyan pair)
        node(fill: cyan-pale, edge: cyan, dash: "dashed")[`self_tok`\ 8-d],
        node(fill: cyan-pale, edge: cyan, dash: "dashed")[`ally_tok`\ 2 × 8-d],
        node(fill: TOK)[`enemy_tok`\ 2 × 8-d],
        node(fill: TOK)[`zone_tok`\ 3 × 6-d],

        arrowdown(col: cyan, dashed: true), arrowdown(col: cyan, dashed: true),
        arrowdown(), arrowdown(),

        // ---- layer 1 : per-set encoders
        node[Enc\ $arrow.r$ 64], node[Enc\ $arrow.r$ 64],
        node[Enc\ $arrow.r$ 64], node[Enc\ $arrow.r$ 64],

        arrowdown(col: cyan, dashed: true), arrowdown(col: cyan, dashed: true),
        arrowdown(), arrowdown(),

        // ---- layer 2 : intra-set self-attention (self passes through)
        node(fill: rgb("#F3F6F9"), size: 17pt)[
          `self_emb`\ #text(fill: muted)[(pass-through)]
        ],
        grid.cell(colspan: 3, node(size: 19pt)[
          *Intra-Set Self-Attention* #h(6pt)
          #text(size: 16pt, fill: muted)[per set · 2 heads · residual + LN]
        ]),

        arrowdown(col: cyan, dashed: true),
        grid.cell(colspan: 3, arrowdown()),

        // ---- layer 3 : cross-attention — the exposed routing channel
        grid.cell(colspan: 4, block(
          width: 100%, fill: cyan-pale, radius: 6pt,
          stroke: (paint: cyan, thickness: 3.2pt),
          inset: (x: 10pt, y: 14pt),
          align(center)[
            #text(size: 21pt, weight: 800, fill: navy)[CROSS-ATTENTION]
            #linebreak()
            #text(size: 17.5pt, fill: navy-soft)[
              Q ← `self_emb` (carries $r_i$) #h(10pt)·#h(10pt)
              K, V ← entity set (carries $r_j$)
            ]
          ],
        )),

        grid.cell(colspan: 4, arrowdown()),

        // ---- layer 4 : head
        grid.cell(colspan: 4, node(size: 18.5pt)[
          Concat `[ally_ctx | enemy_ctx | zone_ctx | self_emb]`
          #h(6pt) $arrow.r$ #h(6pt) MLP(256) #h(6pt) $arrow.r$ #h(6pt) $pi(a | o)$
        ]),
      )

      #v(14pt)
      #align(center, text(size: 17pt, fill: muted)[
        #box(baseline: 2pt,
          line(length: 34pt, stroke: (paint: cyan, thickness: 3pt, dash: "dashed")))
        #h(6pt) role one-hot enters *both* Q and K/V — the load-bearing placement
      ])
    ]

    #v(12pt)

    // ------------------------------------------------------------- Formulation
    #card("Formulation", accent: navy-soft)[
      #set par(justify: false)
      #show math.equation: set text(size: 22pt)

      *Role-conditioned routing.* Because the role one-hot $r_i$ is a sub-block of
      $x_i$ *and* $r_j$ of $x_j$, the attention score is a direct function of the
      *role pair*:

      #block(width: 100%, fill: white, radius: 6pt, stroke: 1pt + border-col,
        inset: (x: 10pt, y: 16pt))[
        $ alpha_(i j) = op("softmax")_j
          ( ((W_Q x_i)^top (W_K x_j)) / sqrt(d_k) ) $
      ]

      #v(10pt)
      *Role-routing matrix.* Entry $(rho, sigma)$ averages the self-query's
      cross-attention over ally slots occupied by role $sigma$; the diagonal is
      zero by construction.

      #block(width: 100%, fill: white, radius: 6pt, stroke: 1pt + border-col,
        inset: (x: 10pt, y: 16pt))[
        $ M_(rho sigma) = 1 / abs(S_sigma) sum_(j in S_sigma) alpha_(i j)
          , quad i : r_i = rho $
        #v(8pt)
        $ "concentration" = "std" ( { M_(rho sigma) }_(rho eq.not sigma) ) $
      ]

      #v(10pt)
      *Formation sensitivity.* Maximum shift in ally self-attention between the
      two fixed probe scenarios — *Clustered* vs. *Spread*:

      #block(width: 100%, fill: white, radius: 6pt, stroke: 1pt + border-col,
        inset: (x: 10pt, y: 16pt))[
        $ Delta_max^((rho)) = max_(s in cal(S)) abs(a_s^"clust" - a_s^"spread") $
      ]

      #v(12pt)
      #text(size: 19pt)[
        *Training.* MAPPO (PPO + centralized critic), 2M steps, 5 seeds, and
        *identical hyperparameters across all five conditions* — the comparison is
        structural, not tuned. GAE $lambda = 0.95$, $gamma = 0.99$, clip $0.2$,
        entropy $0.01$, Adam $3 times 10^(-4)$, batch $512$. Reward mixing
        $r = 0.8 thin r_"ind" + 0.2 thin macron(r)_"team"$.
      ]
    ]

    #v(12pt)

    // ---------------------------------------------------------- Key components
    #card("4. Key Components")[
      #set par(justify: false, leading: 0.58em)

      #let component(title, tag, body) = block(
        width: 100%, fill: white, radius: 6pt,
        stroke: (left: 5pt + cyan, rest: 1pt + border-col),
        inset: (x: 14pt, y: 13pt),
      )[
        #text(size: 22pt, weight: 800, fill: navy, title)
        #h(8pt) #text(size: 16pt, fill: muted, weight: 600, tag)
        #v(7pt)
        #body
      ]

      #stack(
        spacing: 13pt,

        component[Intra-Set Self-Attention][enriches the Key][
          _"Who else is on my side?"_ Each ally token is contextualized against
          the *other* ally tokens before it is consulted as a Key. Residual +
          LayerNorm; `key_padding_mask` with slot-0 force-unmask, preventing NaN
          when a set is fully padded.
          #v(6pt)
          #text(fill: navy, weight: 600)[Ablation (C2):] removing it *preserves*
          routing concentration (entry std $0.286$) but *flips which pair is
          selected* — C2 routes Strike→Vanguard in 4/5 seeds, opposite to both C1
          and the role-semantic prior.
          #v(5pt)
          #text(size: 18pt, fill: muted)[
            ⇒ cross-attention supplies *concentration*; intra-set blocks bias
            *selection*.
          ]
        ],

        component[Cross-Attention Routing][exposes the convention][
          _"Given my role, whose state do I condition on?"_ The self-embedding
          queries each entity set. Because the role one-hot sits in *both* Q and
          K/V, the resulting weight is indexable by $(r_i, r_j)$ — the 3×3 matrix
          falls out directly.
          #v(6pt)
          #text(fill: navy, weight: 600)[Slot-invariance:] the policy is invariant
          to ally count by construction, so a 3v3 checkpoint runs unmodified at
          9v9. Slot-ordering bias is ruled out by reversal.
        ],

        component[Occlusion — the unified metric][apples-to-apples][
          Attention weights and MLP gradients are *not* directly comparable. We
          therefore zero each ally token (mask bit held at 1) and measure policy
          KL from baseline — well-defined for *both* architectures.
          #v(6pt)
          #text(fill: navy, weight: 600)[C1:] occlusion ↔ attention at
          row-normalized Pearson $r = 0.976$.
          #linebreak()
          #text(fill: navy, weight: 600)[C3 @ 9v9:] erratic, one-to-two orders
          larger (max-row KL $≈ 5.6$ vs. $≈ 0.08$); the dominant slot varies by
          seed.
          #v(5pt)
          #text(size: 18pt, fill: muted)[
            ⇒ the MLP is slot-sensitive, but not *role*-structured.
          ]
        ],
      )
    ]
  ],

  // ===========================================================================
  //  COLUMN 3 — Evaluation & Conclusion
  // ===========================================================================
  [
    // --------------------------------------------------------------- Results
    #card("5. Experiments & Results")[
      #set par(justify: false)

      // ---------------- (a) Track 1 win rate --------------------------------
      #text(size: 22pt, weight: 800, fill: navy)[(a) Track 1 — Win rate, SMACv2 3v3]
      #h(1fr)
      #text(size: 17pt, fill: muted)[mean ± std, 5 seeds]
      #v(12pt)

      #stack(
        spacing: 11pt,
        winbar("C1: Full",             0.794, navy,            "0.794 ± .033"),
        winbar("C2: No Self-Attn",     0.656, navy-soft,       "0.656 ± .104"),
        winbar("C4: No Role (shared)", 0.654, rgb("#7A8CA0"),  "0.654 ± .087"),
        winbar("C3: Role + MLP",       0.480, rgb("#A9B4C0"),  "0.480 ± .155"),
        winbar("C5: Flat MLP",         0.464, rgb("#C2CAD3"),  "0.464 ± .049"),
      )

      #v(14pt)
      #callout(fill: cyan-pale)[
        #text(size: 19pt)[
          *Attention helps:* C1 − C3 = *+31.4 pp*. #h(10pt)
          *Intra-set context helps:* C2 trails C1 by *13.8 pp*. #h(10pt)
          *Labels alone do not:* C3 vs. C5 = $0.480$ vs. $0.464$, statistically
          indistinguishable — a role one-hot in an MLP does not by itself produce
          useful role-conditioned routing.
        ]
      ]

      #v(20pt)
      #line(length: 100%, stroke: 1pt + border-col)
      #v(16pt)

      // ---------------- (b) 2x2 results grid --------------------------------
      #text(size: 22pt, weight: 800, fill: navy)[(b) Scaling, transfer & routing structure]
      #v(12pt)

      #let panel(tag, body) = block(
        width: 100%, fill: white, radius: 6pt, stroke: 1.4pt + border-col,
        inset: (x: 11pt, y: 11pt),
      )[
        #text(size: 17.5pt, weight: 700, fill: cyan, tracking: 0.8pt, upper(tag))
        #v(8pt)
        #body
      ]

      #grid(
        columns: (1fr, 1fr),
        gutter: 12pt,

        // -- quadrant 1 : scaling
        panel("Scaling 3v3 → 9v9")[
          #set text(size: 17pt)
          #table(
            columns: (auto, 1fr, 1fr),
            align: (left, center, center),
            stroke: none,
            inset: (x: 3pt, y: 6pt),
            toprule,
            text(weight: 700)[Scale],
            text(weight: 700)[C1 AUC],
            text(weight: 700)[C3 AUC],
            midrule,
            [3v3], strong[.678 ± .019], [.328 ± .059],
            [6v6], strong[.379 ± .064], [.311 ± .098],
            [9v9], strong[.119 ± .067], [.089 ± .052],
            bottomrule,
          )
          #v(6pt)
          #text(size: 15.5pt, fill: muted)[
            6v6 / 9v9 gaps fall inside seed std — *no significance claimed.* The
            *structural* signature persists where win-rate does not.
          ]
        ],

        // -- quadrant 2 : zero-shot transfer
        panel("Zero-shot transfer (3v3-trained)")[
          #set text(size: 17pt)
          #table(
            columns: (auto, 1fr, 1fr, auto),
            align: (left, center, center, right),
            stroke: none,
            inset: (x: 3pt, y: 6pt),
            toprule,
            text(weight: 700)[Target],
            text(weight: 700)[C1],
            text(weight: 700)[C3],
            text(weight: 700)[Gap],
            midrule,
            [3v3], strong[.796], [.468], [32.8 pp],
            [6v6], strong[.212], [.084], [12.8 pp],
            [9v9], strong[.224], [.084], [14.0 pp],
            bottomrule,
          )
          #v(6pt)
          #text(size: 15.5pt, fill: muted)[
            C1 @ 9v9 zero-shot (*.224*) *beats from-scratch* 9v9 C1 (.110). C3
            flatlines at ≈ .084 — MLP weights memorize a fixed slot pattern.
          ]
        ],

        // -- quadrant 3 : the headline result
        grid.cell(colspan: 2, panel("Cond. 1 role-routing matrix — 5 seeds")[
          #grid(
            columns: (110pt, 1fr, 1fr, 1fr),
            gutter: 6pt,
            align: center + horizon,

            text(size: 16pt, fill: muted, weight: 700)[agent \\ ally],
            text(size: 19pt, weight: 700, fill: navy)[Strike],
            text(size: 19pt, weight: 700, fill: navy)[Vanguard],
            text(size: 19pt, weight: 700, fill: navy)[Support],

            text(size: 19pt, weight: 700, fill: navy)[Strike],
            deadcell, heatcell(0.473), heatcell(0.527),

            text(size: 19pt, weight: 700, fill: navy)[Vanguard],
            heatcell(0.565), deadcell, heatcell(0.435),

            text(size: 19pt, weight: 700, fill: navy)[Support],
            heatcell(0.633), heatcell(0.367), deadcell,
          )
          #v(10pt)
          #grid(columns: (1fr, 1fr), gutter: 14pt,
            text(size: 16pt, fill: muted)[
              *Strike → Support* is Strike's largest entry (P4 ✓, 4/5 seeds).
              Concentrated, role-specific, stably off-diagonal.
            ],
            text(size: 16pt, fill: muted)[
              Vanguard *and* Support both route toward *Strike* — the two non-DPS
              roles attend to the engagement-driver.
            ],
          )
        ]),

        // -- quadrant 4 : concentration + Δmax
        grid.cell(colspan: 2, panel("Concentration & formation sensitivity")[
          #grid(columns: (1.05fr, 1fr), gutter: 16pt, align: top,
            [
              #set text(size: 17.5pt)
              #table(
                columns: (1fr, auto),
                align: (left, right),
                stroke: none,
                inset: (x: 3pt, y: 5.5pt),
                toprule,
                text(weight: 700)[Routing entry std], text(weight: 700)[value],
                midrule,
                [C1 attention (SMACv2)],   strong[0.246],
                [C2 no self-attn],         [0.286],
                [C1 attention (MiniGrid)], [0.252],
                [C3 MLP gradient],         text(fill: warn-rust, weight: 700)[0.055],
                midrule,
                text(weight: 700)[C1 / C3 ratio],
                text(weight: 700, fill: cyan)[4.5×],
                bottomrule,
              )
              #v(7pt)
              #text(size: 15.5pt, fill: muted)[
                Row-entropy and KL-from-uniform give the same ranking. MLP
                spatial/role gradient ratio: 5.8× / 2.3× / 3.7×.
              ]
            ],
            [
              #set text(size: 17.5pt)
              #text(weight: 700, fill: navy)[$Delta_max$ by role (Cond. 1)]
              #v(10pt)
              #stack(spacing: 9pt,
                winbar("Support",  0.074 / 0.09, navy,           "0.074", lw: 105pt, nw: 62pt, ls: 17pt),
                winbar("Vanguard", 0.066 / 0.09, navy-soft,      "0.066", lw: 105pt, nw: 62pt, ls: 17pt),
                winbar("Strike",   0.024 / 0.09, rgb("#B7C1CB"), "0.024", lw: 105pt, nw: 62pt, ls: 17pt),
              )
              #v(9pt)
              #text(size: 15.5pt, fill: muted)[
                Support ≈ Vanguard (gap $0.008$, below either std) → *P3 TIED*.
                Strike is clearly formation-*insensitive*, exactly as designed.
              ]
            ],
          )
        ]),
      )
    ]

    #v(12pt)

    // ------------------------------------------------------------- Conclusion
    #card("6. Conclusion & Future Work", accent: navy-soft)[
      #set list(marker: text(fill: cyan, size: 22pt)[▸], spacing: 0.75em)

      - *An architecture-exposed routing channel makes the learned convention
        measurable.* Label-conditioned attention yields a 4.5× more concentrated,
        role-specific signature than MLP baselines.
      - *Structure is more robust than performance.* Win-rate gaps dissolve into
        seed noise at 9v9; the routing signature, the zero-shot transfer
        advantage, and padding-invariance do not.
      - *Seed-aware structural evaluation is mandatory.* Two of four earlier
        "strategic divergences" were small-$n$ artifacts. A 3-seed routing claim
        is not a claim.

      #v(12pt)
      #callout(rail: tie-amber, fill: rgb("#FFF6E8"))[
        #text(size: 19pt)[
          *What this is not.* Attention weights are *not* causal explanations; we
          report them as architecture-exposed patterns, corroborated by gradient
          and occlusion attribution. We do *not* claim the learned convention is
          an equilibrium, nor optimal under any solution concept. The contribution
          is a *measurement procedure.*
        ]
      ]

      #v(12pt)
      *Future work.* Pair the routing pathway with *learned* role discovery
      (e.g. RODE) for deployments without a designer schema; move beyond the
      three-role / Terran restriction; scale the diagnostic to agentic systems
      coordinating in open-ended environments, where strategic structure — rather
      than performance alone — may govern safety and interpretability.
    ]

    #v(12pt)

    // -------------------------------------------- References + Acknowledgement
    #block(
      width: 100%, fill: white, radius: R, stroke: 1.2pt + border-col,
      inset: (x: 18pt, y: 15pt),
    )[
      #set text(size: 15.5pt, fill: muted)
      #set par(justify: false, leading: 0.5em)

      #grid(columns: (1.35fr, 1fr), gutter: 20pt, align: top,
        [
          #text(size: 18pt, weight: 800, fill: navy)[7. References]
          #v(7pt)
          [1] Rashid et al. *QMIX: Monotonic Value Function Factorisation for Deep
          Multi-Agent RL.* ICML 2018.
          #linebreak()
          [2] Yu et al. *The Surprising Effectiveness of PPO in Cooperative
          Multi-Agent Games.* NeurIPS 2022.
          #linebreak()
          [3] Ellis et al. *SMACv2: An Improved Benchmark for Cooperative
          Multi-Agent RL.* NeurIPS D&B 2023.
          #linebreak()
          [4] Wang et al. *RODE: Learning Roles to Decompose Multi-Agent Tasks.*
          ICLR 2021.
          #linebreak()
          [5] Jain & Wallace. *Attention is not Explanation.* NAACL 2019.
        ],
        [
          #text(size: 18pt, weight: 800, fill: navy)[Acknowledgements]
          #v(7pt)
          The author thanks *Prof. Jinseok Seo* (Dong-Eui University) and
          *Gyuyeol Jeong* (Netmarble) for insightful discussions and valuable
          comments that greatly improved this work.
          #v(10pt)
          #text(fill: navy, weight: 600)[
            `yoosunghong.main@gmail.com`
            #h(8pt)·#h(8pt)
            `github: yoosung.dev`
          ]
        ],
      )
    ]
  ],
)
