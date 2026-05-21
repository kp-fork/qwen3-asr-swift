import Foundation
import ArgumentParser
import MAGNeTMusicGen
import AudioCommon

public struct ComposeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "compose",
        abstract: "Generate 30s of music from a text prompt using MAGNeT (MLX, on-device)"
    )

    @Argument(help: "Text prompt describing the music to generate (e.g. \"happy rock\")")
    public var prompt: String

    @Option(name: .shortAndLong, help: "Output WAV path (32 kHz mono)")
    public var output: String = "magnet.wav"

    @Option(
        name: .long,
        help: "Model variant: small-int4 | small-int8 | medium-int4 | medium-int8"
    )
    public var variant: String = "small-int4"

    @Option(name: .long, help: "Sampling temperature (annealed linearly per stage)")
    public var temperature: Float = 3.0

    @Option(name: .long, help: "Top-p (nucleus) sampling threshold")
    public var topP: Float = 0.9

    @Option(name: .long, help: "Max classifier-free guidance coefficient")
    public var cfgMax: Float = 10.0

    @Option(name: .long, help: "Min classifier-free guidance coefficient")
    public var cfgMin: Float = 1.0

    @Option(
        name: .long,
        help: "Comma-separated decoding steps per codebook (default: 20,10,10,10)"
    )
    public var steps: String = "20,10,10,10"

    @Option(name: .long, help: "Random seed for reproducibility")
    public var seed: UInt64?

    public init() {}

    public func run() throws {
        try runAsync {
            guard let variantEnum = MAGNeTVariant(rawValue: variant) else {
                throw ValidationError("Unknown variant '\(variant)'. Use one of: \(MAGNeTVariant.allCases.map { $0.rawValue }.joined(separator: ", "))")
            }
            let decodingSteps = steps.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard decodingSteps.count == 4 else {
                throw ValidationError("--steps must be 4 comma-separated integers (one per codebook), got \(decodingSteps.count)")
            }

            print("Loading MAGNeT \(variant)…")
            let model = try await MAGNeTMusicGen.fromPretrained(
                variant: variantEnum,
                progressHandler: { reportProgress($0, "downloading") })

            print("Prompt: \"\(prompt)\"")
            let params = MAGNeTGenerationParams(
                decodingSteps: decodingSteps,
                maxCfgCoef: cfgMax, minCfgCoef: cfgMin,
                temperature: temperature, topP: topP,
                annealTemp: true, seed: seed)

            print("Generating \(model.config.segmentDuration)s of audio…")
            let start = Date()
            let pcm = model.generate(text: prompt, params: params)
            let elapsed = Date().timeIntervalSince(start)
            let audioSec = Double(pcm.count) / Double(model.config.sampleRate)
            let rtf = elapsed / audioSec
            print("  Generated \(pcm.count) samples (\(String(format: "%.2f", audioSec))s @ \(model.config.sampleRate) Hz)")
            print("  Wall: \(String(format: "%.2f", elapsed))s  RTF: \(String(format: "%.2f", rtf))")

            let outURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: pcm, sampleRate: model.config.sampleRate, to: outURL)
            print("  Saved: \(outURL.path)")
        }
    }
}
