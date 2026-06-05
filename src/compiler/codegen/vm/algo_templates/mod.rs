//! Algorithm template instances — GEMM strategy variants (SPEC 27 REQ-AT-004)
//!
//! 5 GEMM templates: Naive, BLIS, AMX Tile, GPU Tiled, GPU Pipelined.
//! Each template is pure static data (`AlgoTemplate`), consumed by the
//! template interpreter (AT-006) to produce `Vec<TraceOp>`.

pub mod attention_norm_rope_moe;
pub mod sampling;

use crate::compiler::codegen::vm::algo_template::*;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GEMM NAIVE — 三重嵌套循环，无分块 (CPU baseline)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub static GEMM_NAIVE: AlgoTemplate = AlgoTemplate {
    name: "GEMM_NAIVE",
    strategy: AlgoStrategy::GemmNaive,
    device_req: DeviceReq::CpuAny,
    steps: &[
        AlgoStep::Loop {
            bound: "m",
            step: "mr",
            body: &[
                AlgoStep::Loop {
                    bound: "n",
                    step: "nr",
                    body: &[
                        AlgoStep::MicroKernel,
                        AlgoStep::StoreResult {
                            rows_param: "mr",
                            cols_param: "nr",
                        },
                    ],
                },
            ],
        },
    ],
    params: &[
        ("m", AlgoParam::FromGraph("m")),
        ("n", AlgoParam::FromGraph("n")),
        ("mr", AlgoParam::FromDeviceProfile("gemm_mr")),
        ("nr", AlgoParam::FromDeviceProfile("gemm_nr")),
    ],
    micro_kernel: Some(&MicroKernelDef {
        mr: "mr",
        nr: "nr",
        k_step: "1",
        steps: &[
            MicroKernelStep::LoadARow,
            MicroKernelStep::LoadBCol,
            MicroKernelStep::Fma,
        ],
    }),
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GEMM BLIS — MC/NC/KC 三级 cache 分块 (CPU optimized)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub static GEMM_BLIS: AlgoTemplate = AlgoTemplate {
    name: "GEMM_BLIS",
    strategy: AlgoStrategy::GemmBlis,
    device_req: DeviceReq::CpuAvx2,
    steps: &[
        AlgoStep::Loop {
            bound: "mc",
            step: "mc",
            body: &[
                AlgoStep::LoadPanel {
                    matrix: MatrixRole::A,
                    rows_param: "mc",
                    cols_param: "kc",
                },
                AlgoStep::PackBuffer {
                    buffer_name: "packed_a",
                    rows_param: "mc",
                    cols_param: "kc",
                },
                AlgoStep::Loop {
                    bound: "nc",
                    step: "nc",
                    body: &[
                        AlgoStep::LoadPanel {
                            matrix: MatrixRole::B,
                            rows_param: "kc",
                            cols_param: "nc",
                        },
                        AlgoStep::PackBuffer {
                            buffer_name: "packed_b",
                            rows_param: "kc",
                            cols_param: "nc",
                        },
                        AlgoStep::Loop {
                            bound: "mc",
                            step: "mr",
                            body: &[
                                AlgoStep::Loop {
                                    bound: "nc",
                                    step: "nr",
                                    body: &[
                                        AlgoStep::Loop {
                                            bound: "kc",
                                            step: "k_step",
                                            body: &[
                                                AlgoStep::MicroKernel,
                                            ],
                                        },
                                        AlgoStep::StoreResult {
                                            rows_param: "mr",
                                            cols_param: "nr",
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    ],
    params: &[
        ("mc", AlgoParam::FromPressureModel("mc")),
        ("nc", AlgoParam::FromPressureModel("nc")),
        ("kc", AlgoParam::FromPressureModel("kc")),
        ("mr", AlgoParam::FromDeviceProfile("gemm_mr")),
        ("nr", AlgoParam::FromDeviceProfile("gemm_nr")),
        ("k_step", AlgoParam::Const(1)),
    ],
    micro_kernel: Some(&MicroKernelDef {
        mr: "mr",
        nr: "nr",
        k_step: "k_step",
        steps: &[
            MicroKernelStep::LoadARow,
            MicroKernelStep::LoadBCol,
            MicroKernelStep::Fma,
        ],
    }),
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GEMM AMX TILE — Intel AMX tile register (Sapphire Rapids+)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub static GEMM_AMX_TILE: AlgoTemplate = AlgoTemplate {
    name: "GEMM_AMX_TILE",
    strategy: AlgoStrategy::GemmHardwareTile,
    device_req: DeviceReq::CpuAmx,
    steps: &[
        AlgoStep::TileConfig { rows: "16", cols: "16" },
        AlgoStep::Loop {
            bound: "m",
            step: "16",
            body: &[
                AlgoStep::Loop {
                    bound: "n",
                    step: "16",
                    body: &[
                        AlgoStep::Loop {
                            bound: "k",
                            step: "16",
                            body: &[
                                AlgoStep::LoadPanel {
                                    matrix: MatrixRole::A,
                                    rows_param: "16",
                                    cols_param: "16",
                                },
                                AlgoStep::LoadPanel {
                                    matrix: MatrixRole::B,
                                    rows_param: "16",
                                    cols_param: "16",
                                },
                                AlgoStep::TileMma,
                            ],
                        },
                        AlgoStep::StoreResult {
                            rows_param: "16",
                            cols_param: "16",
                        },
                    ],
                },
            ],
        },
        AlgoStep::TileRelease,
    ],
    params: &[
        ("m", AlgoParam::FromGraph("m")),
        ("n", AlgoParam::FromGraph("n")),
        ("k", AlgoParam::FromGraph("k")),
    ],
    micro_kernel: None,
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GEMM GPU TILED — SM80 cp.async + shared memory tiling
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub static GEMM_GPU_TILED: AlgoTemplate = AlgoTemplate {
    name: "GEMM_GPU_TILED",
    strategy: AlgoStrategy::GemmGpuTiled,
    device_req: DeviceReq::GpuSm80,
    steps: &[
        AlgoStep::SharedMemDeclare { name: "smem_a", size_param: "smem_bytes" },
        AlgoStep::SharedMemDeclare { name: "smem_b", size_param: "smem_bytes" },
        AlgoStep::Loop {
            bound: "k",
            step: "bk",
            body: &[
                AlgoStep::AsyncCopyToSmem {
                    buffer_name: "smem_a",
                    size_param: "smem_tile_bytes",
                },
                AlgoStep::AsyncCopyToSmem {
                    buffer_name: "smem_b",
                    size_param: "smem_tile_bytes",
                },
                AlgoStep::AsyncWait { group: 0 },
                AlgoStep::Barrier { barrier_name: "load_done" },
                AlgoStep::Loop {
                    bound: "bm",
                    step: "wm",
                    body: &[
                        AlgoStep::Loop {
                            bound: "bn",
                            step: "wn",
                            body: &[
                                AlgoStep::MicroKernel,
                            ],
                        },
                    ],
                },
            ],
        },
        AlgoStep::StoreResult {
            rows_param: "wm",
            cols_param: "wn",
        },
    ],
    params: &[
        ("bm", AlgoParam::Const(128)),
        ("bn", AlgoParam::Const(128)),
        ("bk", AlgoParam::Const(32)),
        ("wm", AlgoParam::Const(16)),
        ("wn", AlgoParam::Const(16)),
        ("smem_bytes", AlgoParam::Const(16384)),
        ("smem_tile_bytes", AlgoParam::Const(4096)),
    ],
    micro_kernel: Some(&MicroKernelDef {
        mr: "wm",
        nr: "wn",
        k_step: "bk",
        steps: &[
            MicroKernelStep::LoadARow,
            MicroKernelStep::LoadBCol,
            MicroKernelStep::WarpMma,
        ],
    }),
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GEMM GPU PIPELINED — SM90 TMA + wgmma + double buffering
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub static GEMM_GPU_PIPELINED: AlgoTemplate = AlgoTemplate {
    name: "GEMM_GPU_PIPELINED",
    strategy: AlgoStrategy::GemmGpuPipelined,
    device_req: DeviceReq::GpuSm90,
    steps: &[
        AlgoStep::SharedMemDeclare { name: "smem_a_ping", size_param: "smem_half" },
        AlgoStep::SharedMemDeclare { name: "smem_a_pong", size_param: "smem_half" },
        AlgoStep::SharedMemDeclare { name: "smem_b_ping", size_param: "smem_half" },
        AlgoStep::SharedMemDeclare { name: "smem_b_pong", size_param: "smem_half" },
        // Prefetch first tile
        AlgoStep::AsyncCopyToSmem { buffer_name: "smem_a_ping", size_param: "smem_tile_bytes" },
        AlgoStep::AsyncCopyToSmem { buffer_name: "smem_b_ping", size_param: "smem_tile_bytes" },
        AlgoStep::Loop {
            bound: "k",
            step: "bk",
            body: &[
                AlgoStep::AsyncWait { group: 1 }, // wait for previous
                AlgoStep::Barrier { barrier_name: "stage_done" },
                // Compute current tile
                AlgoStep::Loop {
                    bound: "bm",
                    step: "wm",
                    body: &[
                        AlgoStep::MicroKernel,
                    ],
                },
                // Prefetch next tile (overlapped)
                AlgoStep::AsyncCopyToSmem { buffer_name: "smem_a_pong", size_param: "smem_tile_bytes" },
                AlgoStep::AsyncCopyToSmem { buffer_name: "smem_b_pong", size_param: "smem_tile_bytes" },
            ],
        },
        AlgoStep::StoreResult {
            rows_param: "wm",
            cols_param: "wn",
        },
    ],
    params: &[
        ("bm", AlgoParam::Const(128)),
        ("bn", AlgoParam::Const(128)),
        ("bk", AlgoParam::Const(64)),
        ("wm", AlgoParam::Const(16)),
        ("wn", AlgoParam::Const(16)),
        ("smem_half", AlgoParam::Const(32768)),
        ("smem_tile_bytes", AlgoParam::Const(8192)),
    ],
    micro_kernel: Some(&MicroKernelDef {
        mr: "wm",
        nr: "wn",
        k_step: "bk",
        steps: &[
            MicroKernelStep::WarpMma,
        ],
    }),
};

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: GEMM_NAIVE static field values ────────────────────────

    #[test]
    fn gemm_naive_static_fields() {
        // Arrange & Act: read fields from the static
        let name = GEMM_NAIVE.name;
        let strategy = GEMM_NAIVE.strategy;
        let device_req = GEMM_NAIVE.device_req;

        // Assert
        assert_eq!(name, "GEMM_NAIVE");
        assert_eq!(strategy, AlgoStrategy::GemmNaive);
        assert_eq!(device_req, DeviceReq::CpuAny, "NAIVE targets any CPU");
    }

    // ── Test 2: GEMM_NAIVE has exactly 4 params ──────────────────────

    #[test]
    fn gemm_naive_param_count_and_sources() {
        // Arrange
        let params = GEMM_NAIVE.params;

        // Act
        let param_names: Vec<&&str> = params.iter().map(|(name, _)| name).collect();

        // Assert
        assert_eq!(params.len(), 4, "NAIVE has m, n, mr, nr");
        assert!(param_names.contains(&&"m"), "must have 'm' param");
        assert!(param_names.contains(&&"n"), "must have 'n' param");
        assert!(param_names.contains(&&"mr"), "must have 'mr' param");
        assert!(param_names.contains(&&"nr"), "must have 'nr' param");

        // Verify param sources: m/n from graph, mr/nr from device profile
        for (name, param) in params {
            match *name {
                "m" | "n" => assert!(matches!(param, AlgoParam::FromGraph(_)),
                    "{} should be FromGraph", name),
                "mr" | "nr" => assert!(matches!(param, AlgoParam::FromDeviceProfile(_)),
                    "{} should be FromDeviceProfile", name),
                _ => panic!("unexpected param {}", name),
            }
        }
    }

    // ── Test 3: GEMM_BLIS has nested loop structure ──────────────────

    #[test]
    fn gemm_blis_step_structure() {
        // Arrange
        let steps = GEMM_BLIS.steps;

        // Act: the outermost step must be a Loop over mc
        assert_eq!(steps.len(), 1, "BLIS has one outer step");

        if let AlgoStep::Loop { bound, step, body } = &steps[0] {
            // Assert outer loop
            assert_eq!(*bound, "mc");
            assert_eq!(*step, "mc");

            // Body must contain LoadPanel(A), PackBuffer, and a nested Loop over nc
            let has_load_panel_a = body.iter().any(|s| matches!(
                s,
                AlgoStep::LoadPanel { matrix: MatrixRole::A, .. }
            ));
            let has_pack_buffer = body.iter().any(|s| matches!(
                s,
                AlgoStep::PackBuffer { .. }
            ));
            let has_nc_loop = body.iter().any(|s| matches!(
                s,
                AlgoStep::Loop { bound, .. } if *bound == "nc"
            ));

            assert!(has_load_panel_a, "BLIS must load A panel");
            assert!(has_pack_buffer, "BLIS must pack buffer");
            assert!(has_nc_loop, "BLIS must have nc loop");
        } else {
            panic!("outer step must be a Loop");
        }
    }

    // ── Test 4: GEMM_BLIS requires AVX2 ─────────────────────────────

    #[test]
    fn gemm_blis_device_requirement() {
        // Arrange & Act
        let req = GEMM_BLIS.device_req;

        // Assert: BLIS requires at least AVX2
        assert_eq!(req, DeviceReq::CpuAvx2);
        assert!(req.priority() > DeviceReq::CpuAny.priority(),
            "BLIS must have higher priority than CpuAny");
    }

    // ── Test 5: GEMM_AMX_TILE has TileConfig and TileRelease ─────────

    #[test]
    fn gemm_amx_tile_has_config_and_release() {
        // Arrange
        let steps = GEMM_AMX_TILE.steps;

        // Act: find TileConfig and TileRelease
        let has_tile_config = steps.iter().any(|s| matches!(s, AlgoStep::TileConfig { .. }));
        let has_tile_release = steps.iter().any(|s| matches!(s, AlgoStep::TileRelease));

        // Assert
        assert!(has_tile_config, "AMX must start with TileConfig");
        assert!(has_tile_release, "AMX must end with TileRelease");
        assert!(GEMM_AMX_TILE.micro_kernel.is_none(),
            "AMX uses TileMma, no micro-kernel definition");
    }

    // ── Test 6: GEMM_AMX_TILE has no micro_kernel and uses 16x16 tiles

    #[test]
    fn gemm_amx_tile_tile_dimensions() {
        // Arrange
        let steps = GEMM_AMX_TILE.steps;

        // Act: find the TileConfig step
        let tile_config = steps.iter().find_map(|s| match s {
            AlgoStep::TileConfig { rows, cols } => Some((*rows, *cols)),
            _ => None,
        });

        // Assert
        assert_eq!(tile_config, Some(("16", "16")), "AMX uses 16x16 tiles");

        // AMX loop steps use literal "16" for step sizes
        if let AlgoStep::Loop { bound, step, .. } = &steps[1] {
            assert_eq!(*bound, "m");
            assert_eq!(*step, "16");
        } else {
            panic!("second step must be the m-loop");
        }
    }

    // ── Test 7: GEMM_GPU_TILED declares shared memory ────────────────

    #[test]
    fn gemm_gpu_tiled_shared_mem_declare() {
        // Arrange
        let steps = GEMM_GPU_TILED.steps;

        // Act: count SharedMemDeclare steps
        let smem_decls: Vec<&&str> = steps.iter().filter_map(|s| match s {
            AlgoStep::SharedMemDeclare { name, .. } => Some(name),
            _ => None,
        }).collect();

        // Assert: exactly 2 shared memory declarations for smem_a and smem_b
        assert_eq!(smem_decls.len(), 2, "GPU tiled must declare smem_a and smem_b");
        assert!(smem_decls.contains(&&"smem_a"), "must declare smem_a");
        assert!(smem_decls.contains(&&"smem_b"), "must declare smem_b");
    }

    // ── Test 8: GEMM_GPU_TILED params are all Const ──────────────────

    #[test]
    fn gemm_gpu_tiled_params_all_const() {
        // Arrange
        let params = GEMM_GPU_TILED.params;

        // Act & Assert: every param must be Const
        for (name, param) in params {
            match param {
                AlgoParam::Const(val) => {
                    assert!(*val > 0, "param '{}' value {} must be positive", name, val);
                }
                _ => panic!("param '{}' must be Const for GPU tiled, got {:?}", name, param),
            }
        }
        assert_eq!(params.len(), 7, "GPU tiled has 7 const params");
    }

    // ── Test 9: GEMM_GPU_PIPELINED uses double buffering (ping-pong) ─

    #[test]
    fn gemm_gpu_pipelined_double_buffering() {
        // Arrange
        let steps = GEMM_GPU_PIPELINED.steps;

        // Act: collect all SharedMemDeclare names
        let smem_names: Vec<&&str> = steps.iter().filter_map(|s| match s {
            AlgoStep::SharedMemDeclare { name, .. } => Some(name),
            _ => None,
        }).collect();

        // Assert: 4 buffers (ping-pong for A and B)
        assert_eq!(smem_names.len(), 4, "pipelined must have 4 shared mem buffers");
        assert!(smem_names.contains(&&"smem_a_ping"), "must have smem_a_ping");
        assert!(smem_names.contains(&&"smem_a_pong"), "must have smem_a_pong");
        assert!(smem_names.contains(&&"smem_b_ping"), "must have smem_b_ping");
        assert!(smem_names.contains(&&"smem_b_pong"), "must have smem_b_pong");
    }

    // ── Test 10: All 5 templates have unique strategies ──────────────

    #[test]
    fn all_templates_have_unique_strategies() {
        // Arrange
        let templates: &[(&str, AlgoStrategy)] = &[
            ("GEMM_NAIVE", GEMM_NAIVE.strategy),
            ("GEMM_BLIS", GEMM_BLIS.strategy),
            ("GEMM_AMX_TILE", GEMM_AMX_TILE.strategy),
            ("GEMM_GPU_TILED", GEMM_GPU_TILED.strategy),
            ("GEMM_GPU_PIPELINED", GEMM_GPU_PIPELINED.strategy),
        ];

        // Act & Assert: all strategies are distinct
        for i in 0..templates.len() {
            for j in (i + 1)..templates.len() {
                assert_ne!(templates[i].1, templates[j].1,
                    "{} and {} must have different strategies",
                    templates[i].0, templates[j].0);
            }
        }
    }

    // ── Test 11: Templates have ascending device priority ─────────────

    #[test]
    fn templates_device_priority_ascending() {
        // Arrange: templates ordered from least to most specialized
        let cases: &[(&str, DeviceReq)] = &[
            ("GEMM_NAIVE", GEMM_NAIVE.device_req),
            ("GEMM_BLIS", GEMM_BLIS.device_req),
            ("GEMM_AMX_TILE", GEMM_AMX_TILE.device_req),
            ("GEMM_GPU_TILED", GEMM_GPU_TILED.device_req),
            ("GEMM_GPU_PIPELINED", GEMM_GPU_PIPELINED.device_req),
        ];

        // Act & Assert: priorities are strictly ascending
        for i in 0..cases.len() - 1 {
            let pri_cur = cases[i].1.priority();
            let pri_next = cases[i + 1].1.priority();
            assert!(pri_cur < pri_next,
                "{} (priority {}) must be less specialized than {} (priority {})",
                cases[i].0, pri_cur, cases[i + 1].0, pri_next);
        }
    }

    // ── Test 12: GPU templates belong to Gemm family ─────────────────

    #[test]
    fn gpu_templates_are_gemm_family() {
        // Arrange & Act & Assert
        assert_eq!(GEMM_GPU_TILED.strategy.family(), StrategyFamily::Gemm);
        assert_eq!(GEMM_GPU_PIPELINED.strategy.family(), StrategyFamily::Gemm);
        assert_eq!(GEMM_NAIVE.strategy.family(), StrategyFamily::Gemm);
        assert_eq!(GEMM_BLIS.strategy.family(), StrategyFamily::Gemm);
        assert_eq!(GEMM_AMX_TILE.strategy.family(), StrategyFamily::Gemm);
    }

    // ── Test 13: MicroKernel step names are consistent with params ───

    #[test]
    fn micro_kernel_step_names_match_params() {
        // Arrange: collect micro kernel param names from templates that have one
        let templates_with_mk: Vec<(&str, &MicroKernelDef)> = vec![
            ("GEMM_NAIVE", GEMM_NAIVE.micro_kernel.unwrap()),
            ("GEMM_BLIS", GEMM_BLIS.micro_kernel.unwrap()),
            ("GEMM_GPU_TILED", GEMM_GPU_TILED.micro_kernel.unwrap()),
            ("GEMM_GPU_PIPELINED", GEMM_GPU_PIPELINED.micro_kernel.unwrap()),
        ];

        // Act & Assert: micro_kernel mr/nr/k_step must appear in the template params
        for (tmpl_name, mk) in &templates_with_mk {
            let params: Vec<&&str> = match *tmpl_name {
                "GEMM_NAIVE" => GEMM_NAIVE.params.iter().map(|(n, _)| n).collect(),
                "GEMM_BLIS" => GEMM_BLIS.params.iter().map(|(n, _)| n).collect(),
                "GEMM_GPU_TILED" => GEMM_GPU_TILED.params.iter().map(|(n, _)| n).collect(),
                "GEMM_GPU_PIPELINED" => GEMM_GPU_PIPELINED.params.iter().map(|(n, _)| n).collect(),
                _ => unreachable!(),
            };

            assert!(params.contains(&&mk.mr),
                "{}: micro_kernel.mr='{}' must be in params", tmpl_name, mk.mr);
            assert!(params.contains(&&mk.nr),
                "{}: micro_kernel.nr='{}' must be in params", tmpl_name, mk.nr);
            // k_step is either a param name or a numeric literal string
            let k_step_is_param = params.contains(&&mk.k_step);
            let k_step_is_literal = mk.k_step.chars().all(|c| c.is_ascii_digit());
            assert!(k_step_is_param || k_step_is_literal,
                "{}: micro_kernel.k_step='{}' must be a param name or numeric literal",
                tmpl_name, mk.k_step);
        }
    }

    // ── Test 14: GEMM_NAIVE micro_kernel has LoadARow, LoadBCol, Fma steps ─

    #[test]
    fn gemm_naive_micro_kernel_steps() {
        // Arrange
        let mk = GEMM_NAIVE.micro_kernel.expect("NAIVE must have micro_kernel");

        // Act
        let steps = mk.steps;

        // Assert: must contain the canonical 3-step sequence
        assert_eq!(steps.len(), 3, "NAIVE micro kernel has 3 steps");
        assert!(matches!(steps[0], MicroKernelStep::LoadARow),
            "step 0 must be LoadARow");
        assert!(matches!(steps[1], MicroKernelStep::LoadBCol),
            "step 1 must be LoadBCol");
        assert!(matches!(steps[2], MicroKernelStep::Fma),
            "step 2 must be Fma");
    }

    // ── Test 15: GEMM_BLIS params come from PressureModel ─────────────

    #[test]
    fn gemm_blis_blocking_params_from_pressure_model() {
        // Arrange
        let params = GEMM_BLIS.params;

        // Act: find mc, nc, kc params
        let mc = params.iter().find(|(n, _)| *n == "mc");
        let nc = params.iter().find(|(n, _)| *n == "nc");
        let kc = params.iter().find(|(n, _)| *n == "kc");

        // Assert: blocking factors come from PressureModel, not Const
        let mc_param = mc.expect("must have mc param").1;
        let nc_param = nc.expect("must have nc param").1;
        let kc_param = kc.expect("must have kc param").1;

        assert!(matches!(mc_param, AlgoParam::FromPressureModel("mc")),
            "mc must be FromPressureModel(\"mc\")");
        assert!(matches!(nc_param, AlgoParam::FromPressureModel("nc")),
            "nc must be FromPressureModel(\"nc\")");
        assert!(matches!(kc_param, AlgoParam::FromPressureModel("kc")),
            "kc must be FromPressureModel(\"kc\")");
    }

    // ── Test 16: GEMM_GPU_PIPELINED has AsyncWait and Barrier steps ───

    #[test]
    fn gemm_gpu_pipelined_async_wait_and_barrier() {
        // Arrange
        let steps = GEMM_GPU_PIPELINED.steps;

        // Act: find AsyncWait and Barrier inside the k-loop body
        let k_loop = steps.iter().find_map(|s| match s {
            AlgoStep::Loop { bound, .. } if *bound == "k" => Some(s),
            _ => None,
        }).expect("must have k-loop");

        let body = if let AlgoStep::Loop { body, .. } = k_loop {
            body
        } else {
            panic!("k-loop must be a Loop variant");
        };

        let has_async_wait = body.iter().any(|s| matches!(s, AlgoStep::AsyncWait { .. }));
        let has_barrier = body.iter().any(|s| matches!(s, AlgoStep::Barrier { .. }));

        // Assert
        assert!(has_async_wait, "pipelined k-loop must have AsyncWait");
        assert!(has_barrier, "pipelined k-loop must have Barrier");

        // Verify AsyncWait group value
        let wait_group = body.iter().find_map(|s| match s {
            AlgoStep::AsyncWait { group } => Some(*group),
            _ => None,
        }).expect("must find AsyncWait");
        assert_eq!(wait_group, 1, "pipelined uses group=1 for overlap");
    }

    // ── Test 17: GEMM_NAIVE outermost loop is over m with step mr ──────

    #[test]
    fn gemm_naive_outermost_loop_structure() {
        // Arrange
        let steps = GEMM_NAIVE.steps;

        // Assert: exactly one top-level step which is a Loop
        assert_eq!(steps.len(), 1, "NAIVE has one top-level step");

        if let AlgoStep::Loop { bound, step, body } = &steps[0] {
            assert_eq!(*bound, "m", "outer loop iterates over m");
            assert_eq!(*step, "mr", "outer loop steps by mr");

            // Body must contain another Loop over n
            assert_eq!(body.len(), 1, "outer body has one step");
            if let AlgoStep::Loop { bound, step, body: inner_body } = &body[0] {
                assert_eq!(*bound, "n", "inner loop iterates over n");
                assert_eq!(*step, "nr", "inner loop steps by nr");
                assert_eq!(inner_body.len(), 2, "inner body has MicroKernel + StoreResult");
                assert!(matches!(inner_body[0], AlgoStep::MicroKernel));
                assert!(matches!(inner_body[1], AlgoStep::StoreResult { .. }));
            } else {
                panic!("inner step must be a Loop");
            }
        } else {
            panic!("top-level step must be a Loop");
        }
    }

    // ── Test 18: GEMM_AMX_TILE has TileMma inside the k-loop ──────────

    #[test]
    fn gemm_amx_tile_has_tile_mma_in_k_loop() {
        // Arrange
        let steps = GEMM_AMX_TILE.steps;

        // Act: navigate to m-loop → n-loop → k-loop body
        // steps[1] is the m-loop (steps[0] is TileConfig)
        let m_loop = &steps[1];
        if let AlgoStep::Loop { body, .. } = m_loop {
            let n_loop = &body[0];
            if let AlgoStep::Loop { body, .. } = n_loop {
                let k_loop = &body[0];
                if let AlgoStep::Loop { body, .. } = k_loop {
                    // Assert: k-loop body has LoadPanel(A), LoadPanel(B), TileMma
                    assert!(body.iter().any(|s| matches!(
                        s, AlgoStep::LoadPanel { matrix: MatrixRole::A, .. }
                    )), "k-loop must load panel A");
                    assert!(body.iter().any(|s| matches!(
                        s, AlgoStep::LoadPanel { matrix: MatrixRole::B, .. }
                    )), "k-loop must load panel B");
                    assert!(body.iter().any(|s| matches!(s, AlgoStep::TileMma)),
                        "k-loop must have TileMma");
                } else {
                    panic!("expected k-loop");
                }
            } else {
                panic!("expected n-loop");
            }
        } else {
            panic!("expected m-loop");
        }
    }

    // ── Test 19: AlgoParam Derived variant construction and Debug ──────

    #[test]
    fn algo_param_derived_construction_and_variants() {
        // Arrange: construct all 4 ParamArith variants via Derived
        let ceil_div = AlgoParam::Derived { base: "mc", op: ParamArith::CeilDiv, operand: 4 };
        let mul = AlgoParam::Derived { base: "nr", op: ParamArith::Mul, operand: 2 };
        let div = AlgoParam::Derived { base: "kc", op: ParamArith::Div, operand: 8 };
        let max = AlgoParam::Derived { base: "m", op: ParamArith::Max, operand: 1 };
        let min = AlgoParam::Derived { base: "n", op: ParamArith::Min, operand: 64 };

        // Act: format each with Debug
        let ceil_div_dbg = format!("{:?}", ceil_div);
        let mul_dbg = format!("{:?}", mul);
        let div_dbg = format!("{:?}", div);
        let max_dbg = format!("{:?}", max);
        let min_dbg = format!("{:?}", min);

        // Assert: each Debug output mentions the ParamArith variant
        assert!(ceil_div_dbg.contains("CeilDiv"), "must mention CeilDiv");
        assert!(mul_dbg.contains("Mul"), "must mention Mul");
        assert!(div_dbg.contains("Div"), "must mention Div");
        assert!(max_dbg.contains("Max"), "must mention Max");
        assert!(min_dbg.contains("Min"), "must mention Min");

        // Assert: all contain "Derived" and the base param name
        for (label, dbg, base) in [
            ("ceil_div", &ceil_div_dbg, "mc"),
            ("mul", &mul_dbg, "nr"),
            ("div", &div_dbg, "kc"),
            ("max", &max_dbg, "m"),
            ("min", &min_dbg, "n"),
        ] {
            assert!(dbg.contains("Derived"), "{} must contain Derived", label);
            assert!(dbg.contains(base), "{} must contain base '{}'", label, base);
        }
    }

    // ── Test 20: GEMM_GPU_TILED has AsyncCopyToSmem for both buffers ──

    #[test]
    fn gemm_gpu_tiled_async_copy_for_both_buffers() {
        // Arrange
        let steps = GEMM_GPU_TILED.steps;

        // Act: find the k-loop
        let k_loop = steps.iter().find_map(|s| match s {
            AlgoStep::Loop { bound, .. } if *bound == "k" => Some(s),
            _ => None,
        }).expect("must have k-loop");

        let body = if let AlgoStep::Loop { body, .. } = k_loop {
            body
        } else {
            panic!("k-loop must be Loop variant");
        };

        let async_copies: Vec<&&str> = body.iter().filter_map(|s| match s {
            AlgoStep::AsyncCopyToSmem { buffer_name, .. } => Some(buffer_name),
            _ => None,
        }).collect();

        // Assert: async copies for both smem_a and smem_b
        assert_eq!(async_copies.len(), 2, "must have 2 async copies");
        assert!(async_copies.contains(&&"smem_a"), "must async copy to smem_a");
        assert!(async_copies.contains(&&"smem_b"), "must async copy to smem_b");
    }

    // ── Test 21: GEMM_BLIS has LoadPanel for both A and B matrices ────

    #[test]
    fn gemm_blis_load_panel_for_a_and_b() {
        // Arrange
        let steps = GEMM_BLIS.steps;

        // Act: collect all LoadPanel steps recursively from the entire step tree
        fn collect_load_panels(steps: &[AlgoStep]) -> Vec<MatrixRole> {
            let mut roles = Vec::new();
            for step in steps {
                match step {
                    AlgoStep::LoadPanel { matrix, .. } => roles.push(*matrix),
                    AlgoStep::Loop { body, .. } => roles.extend(collect_load_panels(body)),
                    AlgoStep::Seq(body) => roles.extend(collect_load_panels(body)),
                    _ => {}
                }
            }
            roles
        }

        let roles = collect_load_panels(steps);

        // Assert
        assert!(roles.contains(&MatrixRole::A), "BLIS must load panel A");
        assert!(roles.contains(&MatrixRole::B), "BLIS must load panel B");
        assert!(roles.len() >= 2, "BLIS must have at least 2 LoadPanel steps");
    }

    // ── Test 22: GEMM_GPU_PIPELINED micro_kernel uses WarpMma only ────

    #[test]
    fn gemm_gpu_pipelined_micro_kernel_warp_mma_only() {
        // Arrange
        let mk = GEMM_GPU_PIPELINED.micro_kernel.expect("pipelined must have micro_kernel");

        // Assert: only WarpMma step, no LoadARow/LoadBCol/Fma
        assert_eq!(mk.steps.len(), 1, "pipelined MK has exactly 1 step");
        assert!(matches!(mk.steps[0], MicroKernelStep::WarpMma),
            "pipelined MK must be WarpMma only");
    }

    // ── Test 23: All GEMM templates have StoreResult step ──────────────

    #[test]
    fn all_gemm_templates_have_store_result() {
        // Arrange
        let templates: &[(&str, &AlgoTemplate)] = &[
            ("GEMM_NAIVE", &GEMM_NAIVE),
            ("GEMM_BLIS", &GEMM_BLIS),
            ("GEMM_AMX_TILE", &GEMM_AMX_TILE),
            ("GEMM_GPU_TILED", &GEMM_GPU_TILED),
            ("GEMM_GPU_PIPELINED", &GEMM_GPU_PIPELINED),
        ];

        // Act: recursively check each template for StoreResult
        fn has_store_result(steps: &[AlgoStep]) -> bool {
            for step in steps {
                match step {
                    AlgoStep::StoreResult { .. } => return true,
                    AlgoStep::Loop { body, .. } => if has_store_result(body) { return true; },
                    AlgoStep::Seq(body) => if has_store_result(body) { return true; },
                    _ => {}
                }
            }
            false
        }

        // Assert
        for (name, tmpl) in templates {
            assert!(has_store_result(tmpl.steps),
                "{} must contain StoreResult step", name);
        }
    }

    // ── Test 24: GEMM_GPU_TILED has AsyncWait and Barrier in k-loop ───

    #[test]
    fn gemm_gpu_tiled_async_wait_and_barrier_in_k_loop() {
        // Arrange
        let steps = GEMM_GPU_TILED.steps;

        // Act: find the k-loop
        let k_loop = steps.iter().find_map(|s| match s {
            AlgoStep::Loop { bound, .. } if *bound == "k" => Some(s),
            _ => None,
        }).expect("must have k-loop");

        let body = if let AlgoStep::Loop { body, .. } = k_loop {
            body
        } else {
            panic!("k-loop must be Loop variant");
        };

        let has_async_wait = body.iter().any(|s| matches!(s, AlgoStep::AsyncWait { .. }));
        let has_barrier = body.iter().any(|s| matches!(s, AlgoStep::Barrier { .. }));

        // Assert
        assert!(has_async_wait, "GPU tiled k-loop must have AsyncWait");
        assert!(has_barrier, "GPU tiled k-loop must have Barrier");

        // Verify AsyncWait group=0 (synchronous wait)
        let wait_group = body.iter().find_map(|s| match s {
            AlgoStep::AsyncWait { group } => Some(*group),
            _ => None,
        }).expect("must find AsyncWait");
        assert_eq!(wait_group, 0, "tiled uses group=0 for synchronous wait");
    }

    // ── Test 25: Empty steps slice is valid for an AlgoTemplate ────────

    #[test]
    fn empty_steps_template_is_valid() {
        // Arrange: template with zero steps but valid params
        let tmpl = AlgoTemplate {
            name: "empty_steps",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[],
            params: &[("dim", AlgoParam::FromGraph("dim"))],
            micro_kernel: None,
        };

        // Act
        let step_count = tmpl.steps.len();
        let param_count = tmpl.params.len();

        // Assert: empty steps is a valid state (e.g., a passthrough/no-op template)
        assert_eq!(step_count, 0, "empty steps must have zero count");
        assert_eq!(param_count, 1, "params can exist independently of steps");
        assert_eq!(tmpl.name, "empty_steps");
    }

    // ── Test 26: GEMM_NAIVE name matches strategy string ──────────────

    #[test]
    fn gemm_naive_name_matches_strategy_debug() {
        // Arrange
        let name = GEMM_NAIVE.name;
        let strategy_dbg = format!("{:?}", GEMM_NAIVE.strategy);

        // Act: name should contain "NAIVE", strategy debug should contain "GemmNaive"
        let name_upper = name.to_uppercase();

        // Assert
        assert!(name_upper.contains("NAIVE"), "name must contain NAIVE");
        assert!(strategy_dbg.contains("GemmNaive"), "strategy debug must contain GemmNaive");
        assert!(strategy_dbg.contains("Naive"), "strategy debug must mention Naive");
    }

    // ── Test 27: GEMM_GPU_PIPELINED const param value boundaries ──────

    #[test]
    fn gemm_gpu_pipelined_const_params_positive() {
        // Arrange
        let params = GEMM_GPU_PIPELINED.params;

        // Act: extract all Const params and their values
        let const_vals: Vec<(&str, usize)> = params.iter()
            .filter_map(|(name, p)| match p {
                AlgoParam::Const(v) => Some((*name, *v)),
                _ => None,
            })
            .collect();

        // Assert: all const values are positive and power-of-two aligned
        assert!(!const_vals.is_empty(), "pipelined must have const params");
        for (name, val) in &const_vals {
            assert!(*val > 0, "param '{}' value must be positive", name);
            assert!(*val % 2 == 0, "param '{}' value {} should be even", name, val);
        }
    }

    // ── Test 28: GEMM_AMX_TILE params are all FromGraph ───────────────

    #[test]
    fn gemm_amx_tile_params_all_from_graph() {
        // Arrange
        let params = GEMM_AMX_TILE.params;

        // Act & Assert: every param must be FromGraph (AMX has no Const/PressureModel params)
        for (name, param) in params {
            match param {
                AlgoParam::FromGraph(key) => {
                    assert!(!key.is_empty(), "{} param key must not be empty", name);
                }
                _ => panic!(
                    "param '{}' must be FromGraph for AMX, got {:?}",
                    name, param
                ),
            }
        }
        assert_eq!(params.len(), 3, "AMX has m, n, k params");
    }

    // ── Test 29: AlgoParam::Const(0) is distinct from Derived ─────────

    #[test]
    fn algo_param_const_zero_is_not_derived() {
        // Arrange
        let const_zero = AlgoParam::Const(0);
        let derived_min = AlgoParam::Derived {
            base: "x",
            op: ParamArith::Min,
            operand: 0,
        };

        // Act: format both with Debug
        let const_dbg = format!("{:?}", const_zero);
        let derived_dbg = format!("{:?}", derived_min);

        // Assert: different variant names appear in debug output
        assert!(const_dbg.contains("Const"), "must be Const variant");
        assert!(derived_dbg.contains("Derived"), "must be Derived variant");
        assert_ne!(const_dbg, derived_dbg, "Const(0) and Derived must differ");
    }

    // ── Test 30: GEMM_BLIS micro_kernel uses same mr/nr as params ────

    #[test]
    fn gemm_blis_micro_kernel_refs_match_param_names() {
        // Arrange
        let mk = GEMM_BLIS.micro_kernel.expect("BLIS must have micro_kernel");
        let param_names: Vec<&&str> = GEMM_BLIS.params.iter().map(|(n, _)| n).collect();

        // Act: verify mk.mr and mk.nr reference existing param names
        let mr_in_params = param_names.iter().any(|n| **n == mk.mr);
        let nr_in_params = param_names.iter().any(|n| **n == mk.nr);

        // Assert
        assert!(mr_in_params, "micro_kernel.mr='{}' must exist in params", mk.mr);
        assert!(nr_in_params, "micro_kernel.nr='{}' must exist in params", mk.nr);
        assert_eq!(mk.mr, "mr", "BLIS mk.mr must reference 'mr' param");
        assert_eq!(mk.nr, "nr", "BLIS mk.nr must reference 'nr' param");
    }

    // ── Test 31: GEMM_GPU_TILED smem_bytes > smem_tile_bytes ──────────

    #[test]
    fn gemm_gpu_tiled_smem_bytes_exceeds_tile_bytes() {
        // Arrange: extract Const values for smem_bytes and smem_tile_bytes
        let smem_bytes = GEMM_GPU_TILED.params.iter()
            .find(|(n, _)| *n == "smem_bytes")
            .and_then(|(_, p)| match p { AlgoParam::Const(v) => Some(*v), _ => None })
            .expect("smem_bytes must be Const");
        let smem_tile_bytes = GEMM_GPU_TILED.params.iter()
            .find(|(n, _)| *n == "smem_tile_bytes")
            .and_then(|(_, p)| match p { AlgoParam::Const(v) => Some(*v), _ => None })
            .expect("smem_tile_bytes must be Const");

        // Assert: total shared memory must hold at least both tiles (A + B)
        assert!(smem_bytes > smem_tile_bytes,
            "smem_bytes ({}) must exceed smem_tile_bytes ({}) to hold A+B tiles",
            smem_bytes, smem_tile_bytes);
        assert!(smem_bytes >= smem_tile_bytes * 2,
            "smem_bytes ({}) must be at least 2x tile bytes ({}) for A+B buffers",
            smem_bytes, smem_tile_bytes);
    }

    // ── Test 32: DeviceReq::GpuSm80 vs GpuSm90 priorities are distinct ─

    #[test]
    fn gpu_sm80_and_sm90_priorities_are_distinct() {
        // Arrange
        let sm80 = DeviceReq::GpuSm80;
        let sm90 = DeviceReq::GpuSm90;

        // Act
        let p80 = sm80.priority();
        let p90 = sm90.priority();

        // Assert: distinct and ordered
        assert_ne!(p80, p90, "SM80 and SM90 must have distinct priorities");
        assert!(p80 < p90, "SM80 priority ({}) must be less than SM90 ({})", p80, p90);
    }

    // ── Test 33: GEMM_NAIVE has no LoadPanel or PackBuffer steps ──────

    #[test]
    fn gemm_naive_has_no_load_panel_or_pack_buffer() {
        // Arrange
        let steps = GEMM_NAIVE.steps;

        // Act: recursively search for LoadPanel and PackBuffer steps
        fn has_step_kind(steps: &[AlgoStep], check: fn(&AlgoStep) -> bool) -> bool {
            for s in steps {
                if check(s) { return true; }
                if let AlgoStep::Loop { body, .. } = s {
                    if has_step_kind(body, check) { return true; }
                }
                if let AlgoStep::Seq(body) = s {
                    if has_step_kind(body, check) { return true; }
                }
            }
            false
        }

        let has_load_panel = has_step_kind(steps, |s| matches!(s, AlgoStep::LoadPanel { .. }));
        let has_pack_buffer = has_step_kind(steps, |s| matches!(s, AlgoStep::PackBuffer { .. }));

        // Assert: NAIVE is the simplest GEMM, no panel loading or packing
        assert!(!has_load_panel, "NAIVE must not have LoadPanel steps");
        assert!(!has_pack_buffer, "NAIVE must not have PackBuffer steps");
    }

    // ── Test 34: GEMM_GPU_TILED k_step param references exist ─────────

    #[test]
    fn gemm_gpu_tiled_k_loop_references_existing_param() {
        // Arrange: find the k-loop in GPU TILED steps
        let k_loop = GEMM_GPU_TILED.steps.iter().find_map(|s| match s {
            AlgoStep::Loop { bound, step, .. } if *bound == "k" => Some(*step),
            _ => None,
        }).expect("must have k-loop");

        // Act: verify the k-loop step name exists in params
        let param_names: Vec<&&str> = GEMM_GPU_TILED.params.iter().map(|(n, _)| n).collect();

        // Assert: "bk" must be a defined param (the blocking factor for k)
        assert_eq!(k_loop, "bk", "k-loop step must be 'bk'");
        assert!(param_names.iter().any(|n| **n == "bk"),
            "k-loop step 'bk' must exist in template params");
    }
}


