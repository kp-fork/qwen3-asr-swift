import XCTest
import Foundation
import MLX
import MLXNN
import Tokenizers
@testable import MAGNeTMusicGen

/// Step-by-step comparison against Python reference for "happy rock".
/// Python ground truth (mlx 0.31, t5-base bf16, encodec-32khz-fp32):
///
///   token_ids = [1095, 2480, 1]
///   T5 mean=-0.005768 std=0.232194 min=-0.871 max=1.625
///   T5[0,0,:8] = [-0.0649, -0.0576, -0.1211, 0.2852, -0.2461, 0.1582, 0.2451, -0.3223]
///   cond after proj mean=0.005914 std=0.293074
///   cond[0,0,:8] = [0.07413, -0.32806, -0.00565, -0.26058, -0.15961, 0.05370, 0.23865, -0.02777]
///   LM stage=0 cond_logits[0,0,0,:8] = [0.4557, -1.3019, -1.4090, 0.7347, 1.1703, -1.1530, -0.3308, -3.5498]
///   LM stage=0 uncond_logits[1,0,0,:8] = [-0.4370, -3.0242, -3.3304, -1.5964, 0.4973, -2.8707, -2.5039, -6.0367]
///   stage-0 argmax histogram (top): {83: 74.1%, 1103: 25.9%, 1964: 0.1%}
final class E2EDiagnosticsTests: XCTestCase {

    func testTokenizerProducesSameIdsForHappyRock() async throws {
        let paths = try await MAGNeTDownloader.ensureDownloaded(
            variant: .smallInt4, progressHandler: nil)
        let tokenizer = try await AutoTokenizer.from(modelFolder: paths.t5Dir)
        let ids = tokenizer.encode(text: "happy rock")
        print("[SWIFT] token IDs: \(ids)")
        XCTAssertEqual(ids, [1095, 2480, 1],
                       "Tokenizer must produce Python's exact IDs for 'happy rock'")
    }

    func testT5EncoderMatchesPython() async throws {
        let paths = try await MAGNeTDownloader.ensureDownloaded(
            variant: .smallInt4, progressHandler: nil)
        let t5Cfg = try T5ModelConfig.load(
            from: paths.t5Dir.appendingPathComponent("config.json"))
        let raw = try MLX.loadArrays(url: paths.t5Dir.appendingPathComponent("model.safetensors"))
        let weights = T5Encoder.sanitize(raw)

        // Probe what `layers.0.attention.query_proj.weight` looks like in the
        // sanitized dict BEFORE loading. Python's first row is [0.0762, ...]
        if let qpw = weights["layers.0.attention.query_proj.weight"] {
            let arr = qpw.asArray(Float.self)
            print("[SWIFT] sanitized layers.0.attention.query_proj.weight[0,:5] = \(Array(arr.prefix(5)))")
            print("[SWIFT] sanitized layers.0.attention.query_proj.weight.shape = \(qpw.shape)")
        } else {
            print("[SWIFT] !! sanitized dict missing layers.0.attention.query_proj.weight")
            print("[SWIFT] keys containing 'query_proj' or 'q.':")
            for k in weights.keys.sorted() where k.contains("query_proj") || k.contains(".q.") {
                print("    \(k)")
            }
        }

        let encoder = T5Encoder(config: t5Cfg)
        try encoder.loadSanitizedWeights(weights)

        let ids: [Int32] = [1095, 2480, 1]
        let tokens = MLXArray(ids).reshaped([1, ids.count])

        // 1. Embedding sanity check (FP raw values).
        let emb = encoder.wte(tokens)
        eval(emb)
        let embArr = emb.asArray(Float.self)
        print("[SWIFT] wte(ids)[0,0,:8] = \(Array(embArr.prefix(8)))")
        // Python: [2.890625, 7.03125, -9.8125, 10.1875, -21.375, -17.5, 1.3046875, 9.3125]
        XCTAssertEqual(embArr[0], 2.890625, accuracy: 0.01, "wte[0,0,0]")
        XCTAssertEqual(embArr[1], 7.03125, accuracy: 0.01, "wte[0,0,1]")
        XCTAssertEqual(embArr[2], -9.8125, accuracy: 0.01, "wte[0,0,2]")

        // 2. Relative attention bias sanity check.
        let pos = encoder.relativeAttentionBias(queryLength: 3, keyLength: 3)
        eval(pos)
        let posArr = pos.asArray(Float.self)
        print("[SWIFT] pos_bias shape=\(pos.shape) [0,0,:3] = \(Array(posArr.prefix(3)))")
        // Python head 0 query 0: [2.890625, 1.3046875, 1.5078125]
        XCTAssertEqual(posArr[0], 2.890625, accuracy: 0.01, "pos_bias[head=0,q=0,k=0]")

        // 3. Layer-by-layer dump (compare to Python layer-by-layer stats).
        var x = encoder.wte(tokens)
        let bias = encoder.relativeAttentionBias(queryLength: 3, keyLength: 3)
        eval(x)
        eval(bias)

        // Probe: dump layer 0's ln1(x), q after queryProj, attention output.
        let l0 = encoder.layers[0]
        let xn = l0.ln1(x)
        eval(xn)
        let xnArr = xn.asArray(Float.self)
        print("[SWIFT] layer0 ln1(x)[0,0,:5] = \(Array(xnArr.prefix(5)))")
        let q0 = l0.attention.queryProj(xn)
        eval(q0)
        let qArr = q0.asArray(Float.self)
        print("[SWIFT] layer0 queryProj(ln1)[0,0,:5] = \(Array(qArr.prefix(5)))")
        // Also dump the actual queryProj.weight first row
        let qw = l0.attention.queryProj.weight
        let qwArr = qw.asArray(Float.self)
        print("[SWIFT] layer0 queryProj.weight[0,:5] = \(Array(qwArr.prefix(5)))")
        print("[SWIFT] layer0 queryProj.weight.shape = \(qw.shape)")
        for (i, layer) in encoder.layers.enumerated() {
            x = layer(x, bias: bias)
            eval(x)
            if i < 3 || i == 11 {
                let a = x.asArray(Float.self)
                let m = a.reduce(0, +) / Float(a.count)
                let s = (a.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Float(a.count)).squareRoot()
                // a is [B=1, T=3, D=768] flat; index [0,0,0]=a[0], [0,0,3]=a[3]
                print(String(format: "[SWIFT] layer %d: mean=%.6f std=%.6f [0,0,0]=%.6f [0,0,3]=%.6f",
                             i, m, s, a[0], a[3]))
            }
        }
        let out = encoder.ln(x)
        eval(out)
        let flat = out.asArray(Float.self)
        let n = Float(flat.count)
        let mean = flat.reduce(0, +) / n
        let std = (flat.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / n).squareRoot()
        print("[SWIFT] T5 final stats: mean=\(mean) std=\(std)")
        print("[SWIFT] T5[0,0,:8] = \(Array(flat.prefix(8)))")
        XCTAssertEqual(mean, -0.005768, accuracy: 0.005, "T5 mean")
        XCTAssertEqual(std, 0.232194, accuracy: 0.005, "T5 std")
        XCTAssertEqual(flat[0], -0.0649, accuracy: 0.005, "T5[0,0,0]")
        XCTAssertEqual(flat[3], 0.2852, accuracy: 0.005, "T5[0,0,3]")
    }

    func testLMFirstForwardMatchesPython() async throws {
        let model = try await MAGNeTMusicGen.fromPretrained(variant: .smallInt4)

        // Reproduce the exact Python probe: tokenize 'happy rock', T5 encode,
        // project to LM dim, build CFG batch, run LM forward at stage=0 on
        // an all-mask input.
        let ids: [Int32] = [1095, 2480, 1]
        let tokenArray = MLXArray(ids).reshaped([1, ids.count])

        let t5Out = model._t5(tokenArray)
        let cond = MLX.matmul(t5Out, model._textProjWeight.T) + model._textProjBias
        let uncond = MLXArray.zeros(cond.shape, dtype: cond.dtype)
        let conditioning = MLX.concatenated([cond, uncond], axis: 0)
        let T = model.config.seqLen
        let K = model.config.nQ
        let maskId = model.config.maskTokenId
        var gen = MLXArray.full([1, K, T], values: MLXArray(Int32(maskId)))
        let lmInput = MLX.concatenated([gen, gen], axis: 0).transposed(0, 2, 1)
        let allLogits = model._lm(lmInput, conditioning: conditioning, stage: 0)
        eval(allLogits)

        // Print stats matching the Python probe.
        let arr = allLogits.asArray(Float.self)
        // allLogits shape = [2, T, K, card]
        let card = model.config.card
        let cardOffset0 = 0 * card                                  // (b=0, t=0, k=0)
        let cardOffsetUncond0 = (1 * T * K + 0 * K + 0) * card      // (b=1, t=0, k=0)
        print("[SWIFT] LM cond_logits[0,0,0,:8]   = \(Array(arr[cardOffset0..<(cardOffset0+8)]))")
        print("[SWIFT] LM uncond_logits[1,0,0,:8] = \(Array(arr[cardOffsetUncond0..<(cardOffsetUncond0+8)]))")

        // Argmax token histogram of stage-0 cond.
        var argmaxByT: [Int] = Array(repeating: 0, count: T)
        for t in 0..<T {
            let base = 0 * (T * K * card) + t * (K * card) + 0 * card
            var best = -Float.infinity
            var bestIdx = -1
            for c in 0..<card {
                let v = arr[base + c]
                if v > best { best = v; bestIdx = c }
            }
            argmaxByT[t] = bestIdx
        }
        var hist: [Int: Int] = [:]
        for v in argmaxByT { hist[v, default: 0] += 1 }
        let top = hist.sorted { $0.value > $1.value }.prefix(5)
        print("[SWIFT] stage-0 argmax histogram (top 5):")
        for (tok, count) in top {
            print("    token \(tok): \(count) times (\(Double(count) / Double(T) * 100)%)")
        }

        // Spot-check first logit values vs Python.
        XCTAssertEqual(arr[0], 0.4557, accuracy: 0.05, "cond_logits[0,0,0,0]")
        XCTAssertEqual(arr[3], 0.7347, accuracy: 0.05, "cond_logits[0,0,0,3]")
        XCTAssertEqual(arr[cardOffsetUncond0], -0.4370, accuracy: 0.05,
                       "uncond_logits[1,0,0,0]")
    }
}
