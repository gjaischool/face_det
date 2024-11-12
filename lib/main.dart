import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:audioplayers/audioplayers.dart';

void main() async {
  // Flutter 엔진 초기화
  WidgetsFlutterBinding.ensureInitialized();
  // 사용 가능한 카메라 목록 가져오기
  final cameras = await availableCameras();
  // 앱 실행
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // 메인 화면 졸음감지 위젯
      home: DrowsinessDetection(cameras: cameras),
    );
  }
}

class DrowsinessDetection extends StatefulWidget {
  final List<CameraDescription> cameras;

  const DrowsinessDetection({super.key, required this.cameras});

  @override
  _DrowsinessDetectionState createState() => _DrowsinessDetectionState();
}

class _DrowsinessDetectionState extends State<DrowsinessDetection> {
  late CameraController _controller;
  bool isProcessing = false;
  final audioPlayer = AudioPlayer();
  Interpreter? _interpreter;

  // 이미지 처리를 위한 버퍼
  List<List<List<double>>>? _inputBuffer;

  // 졸음 감지를 위한 상태 변수들
  int drowsyFrameCount = 0;
  static const int DROWSY_FRAME_THRESHOLD = 20;
  static const double DROWSY_THRESHOLD = 0.7;
  DateTime? lastAlertTime;

  @override
  void initState() {
    super.initState();
    _initializeCamera(); // 카메라 초기화
    _loadModel(); // AI 모델 로드
    // 입력 버퍼 초기화
    _inputBuffer = List.generate(
      224,
      (_) => List.generate(
        224,
        (_) => List.filled(3, 0.0),
      ),
    );
  }

  // 카메라 스트림 시작
  Future<void> _initializeCamera() async {
    _controller = CameraController(
      widget.cameras[1],
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await _controller.initialize();

    // 무한 루프. 카메라로 부터 실시간 이미지 스트림 받아옴
    await _controller.startImageStream((CameraImage image) {
      if (!isProcessing) {
        isProcessing = true; //한 프레임 처리 끝나기전에 다른 프레임 처리하지 않도록
        _processCameraImage(image); // 이미지 처리 시작
      }
    });

    setState(() {});
  }

  Future<void> _loadModel() async {
    try {
      final options = InterpreterOptions()..threads = 4; // 멀티스레딩 활성화
      // ..useNnapi = true; // Android Neural Networks API 사용

      _interpreter = await Interpreter.fromAsset(
        'assets/converted_jt_model.tflite',
        options: options,
      );
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_interpreter == null || _inputBuffer == null) return;

    try {
      // YUV420 이미지를 직접 처리
      final int width = image.width;
      final int height = image.height;
      final int uvRowStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = image.planes[1].bytesPerPixel!;

      // YUV420 형식의 카메라 이미지를 RGB로 변환하고 224x224 크기로 리샘플링
      for (int x = 0; x < 224; x++) {
        for (int y = 0; y < 224; y++) {
          int sourceX = (x * width ~/ 224);
          int sourceY = (y * height ~/ 224);

          // YUV 값 추출
          final int uvIndex =
              uvPixelStride * (sourceX ~/ 2) + uvRowStride * (sourceY ~/ 2);
          final int index = sourceY * width + sourceX;

          final yp = image.planes[0].bytes[index];
          final up = image.planes[1].bytes[uvIndex];
          final vp = image.planes[2].bytes[uvIndex];

          // YUV to RGB 변환 및 정규화
          int r = (yp + 1.402 * (vp - 128)).round().clamp(0, 255);
          int g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128))
              .round()
              .clamp(0, 255);
          int b = (yp + 1.772 * (up - 128)).round().clamp(0, 255);

          // 정규화된 값을 입력 버퍼에 저장
          // YUV를 RGB로 변환하고 정규화 (-1 ~ 1 범위로) (MobileNetV2 전처리에 맞춤):
          _inputBuffer![y][x][0] = (r / 127.5) - 1;
          _inputBuffer![y][x][1] = (g / 127.5) - 1;
          _inputBuffer![y][x][2] = (b / 127.5) - 1;
        }
      }

      // 모델 실행을 위한 입력 준비
      final input = [_inputBuffer!];
      final output = List.filled(1, 0.0).reshape([1, 1]); // 이진분류

      // 모델 실행
      _interpreter!.run(input, output);

      // 결과 처리
      final isEyeClosed = output[0] > 0.5;

      if (isEyeClosed) {
        drowsyFrameCount++;
        if (drowsyFrameCount >= DROWSY_FRAME_THRESHOLD) {
          _handleDrowsiness();
        }
      } else {
        drowsyFrameCount = 0;
      }
    } catch (e) {
      print('Error processing image: $e');
    } finally {
      isProcessing = false;
    }
  }

  void _handleDrowsiness() {
    final now = DateTime.now();
    if (lastAlertTime == null ||
        now.difference(lastAlertTime!) > const Duration(seconds: 5)) {
      _showAlert();
      lastAlertTime = now;
      drowsyFrameCount = 0;
    }
  }

  void _showAlert() {
    audioPlayer.play(AssetSource('alert.wav'));

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('졸음이 감지되었습니다! 휴식이 필요합니다.'),
          backgroundColor: Colors.red,
          duration: Duration(seconds: 3),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Container();
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text('실시간 졸음 감지'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info),
            onPressed: () => _showStatusDialog(),
          ),
        ],
      ),
      body: Stack(
        children: [
          CameraPreview(_controller),
          Positioned(
            top: 10,
            right: 10,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                'FPS: 30\n감지 상태: ${isProcessing ? "처리 중" : "대기"}',
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _showStatusDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('상태 정보'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Text('눈 감김 확률: ${(output[0] * 100).toStringAsFixed(1)}%'),
            Text('연속 프레임: $drowsyFrameCount'),
            Text('임계값: ${(DROWSY_THRESHOLD * 100).toStringAsFixed(1)}%'),
            const Text('알림 간격: 5초'),
            Text('마지막 알림: ${lastAlertTime?.toString() ?? "없음"}'),
          ],
        ),
        actions: [
          TextButton(
            child: const Text('확인'),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    _interpreter?.close();
    super.dispose();
  }
}
