import Foundation
#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#if canImport(UniformTypeIdentifiers)
import UniformTypeIdentifiers
#endif

/// 対数 Mel スペクトログラムを PNG 画像としてレンダリングする高精度可視化レンダラー
public enum SpectrogramRenderer {

    /// 対数 Mel スペクトログラムからカラー PNG 画像を生成してファイルに保存する
    public static func renderMelToPNG(
        mel: [[Float]],
        outputPath: String,
        scale: Int = 2
    ) throws {
        let frames = mel.count
        let channels = AudioConfig.melChannels // 64
        if frames <= 0 || channels <= 0 {
            return
        }

        // min / max 探索
        var minVal = Float.greatestFiniteMagnitude
        var maxVal = -Float.greatestFiniteMagnitude
        var t = 0
        while t < frames {
            var c = 0
            while c < channels {
                let v = mel[t][c]
                if v < minVal { minVal = v }
                if maxVal < v { maxVal = v }
                c += 1
            }
            t += 1
        }

        let range = max(1e-5, maxVal - minVal)
        let width = frames * scale
        let height = channels * scale

        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        var y = 0
        while y < height {
            let melCh = (height - 1 - y) / scale // 低周波を画像下部に配置
            var x = 0
            while x < width {
                let frameIdx = x / scale
                let v = mel[frameIdx][melCh]
                var norm = (v - minVal) / range
                if norm < 0.0 { norm = 0.0 }
                if 1.0 < norm { norm = 1.0 }

                // Magma / Viridis 風カラーマッピング (暗紫 -> マゼンタ -> 橙 -> 黄)
                let r: UInt8
                let g: UInt8
                let b: UInt8
                switch norm {
                case ..<0.33:
                    let f = norm / 0.33
                    r = UInt8(f * 120.0)
                    g = UInt8(f * 20.0)
                    b = UInt8(60.0 + (f * 100.0))
                case 0.33..<0.66:
                    let f = (norm - 0.33) / 0.33
                    r = UInt8(120.0 + (f * 110.0))
                    g = UInt8(20.0 + (f * 80.0))
                    b = UInt8(160.0 - (f * 120.0))
                default:
                    let f = (norm - 0.66) / 0.34
                    r = UInt8(230.0 + (f * 25.0))
                    g = UInt8(100.0 + (f * 155.0))
                    b = UInt8(40.0 + (f * 180.0))
                }

                let pxOffset = ((y * width) + x) * 4
                pixelData[pxOffset + 0] = r
                pixelData[pxOffset + 1] = g
                pixelData[pxOffset + 2] = b
                pixelData[pxOffset + 3] = 255

                x += 1
            }
            y += 1
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let provider = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            return
        }

        let outURL = URL(fileURLWithPath: outputPath)
        let outDir = outURL.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: outDir.path) != true {
            try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        }

        let destType: CFString = "public.png" as CFString
        guard let destination = CGImageDestinationCreateWithURL(outURL as CFURL, destType, 1, nil) else {
            return
        }
        CGImageDestinationAddImage(destination, cgImage, nil)
        CGImageDestinationFinalize(destination)
    }

    /// WAV ファイルから直接 Mel スペクトログラムを抽出して PNG 保存する
    public static func renderWavToPNG(
        wavPath: String,
        outputPath: String,
        scale: Int = 2
    ) throws {
        let reader = WavAudioReader()
        let pcm = try reader.loadWav16k(from: wavPath)
        let extractor = MelSpectrogramExtractor(
            sampleRate: Float(AudioConfig.sampleRate),
            melChannels: AudioConfig.melChannels
        )
        let mel = extractor.extractLogMel(pcm: pcm)
        try renderMelToPNG(mel: mel, outputPath: outputPath, scale: scale)
    }
}
#endif
