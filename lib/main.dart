import 'dart:collection';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:audioplayers/audioplayers.dart';
import 'dart:math';

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
  DrowsinessDetectionState createState() => DrowsinessDetectionState();
}

class DrowsinessDetectionState extends State<DrowsinessDetection> {
  late CameraController _controller;
  bool isProcessing = false;
  final audioPlayer = AudioPlayer();
  Interpreter? _interpreter;

  // 이미지 처리를 위한 버퍼
  List<List<List<double>>>? _inputBuffer;
  // 출력 버퍼를 24개의 키포인트 좌표를 저장하도록 수정
  final List<double> _outputBuffer = List.filled(24, 0.0);

  // 졸음 감지를 위한 상태 변수들
  int drowsyFrameCount = 0;
  static const int drowsyFrameThreshold = 8;
  static const double earThreshold = 0.26; // EAR 임계값 설정

  // EAR 히스토리 저장을 위한 큐
  final Queue<double> earHistory = Queue<double>();
  static const int historyMaxLength = 30; // 1초(30fps) 동안의 기록 저장

  // 상태 표시를 위한 변수
  String eyeState = "측정 중...";
  Color stateColor = Colors.white;

  // 키포인트를 저장할 리스트 추가
  List<Offset> keypoints = [];

  DateTime? lastAlertTime;

  // EAR 계산 및 상태 업데이트 함수
  void _updateEyeState(double ear) {
    // EAR 히스토리 업데이트
    earHistory.addLast(ear);
    if (earHistory.length > historyMaxLength) {
      earHistory.removeFirst();
    }

    // 현재 상태 판단
    setState(() {
      if (ear < earThreshold) {
        eyeState = "눈 감음";
        stateColor = Colors.red;
        drowsyFrameCount++;

        if (drowsyFrameCount >= drowsyFrameThreshold) {
          _handleDrowsiness();
        }
      } else {
        eyeState = "눈 뜸";
        stateColor = Colors.green;
        drowsyFrameCount = 0;
      }
    });
  }

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
      //..useNnapi = true; // Android Neural Networks API 사용  텐서플로우2.2이상부터 우린 2.14...

      _interpreter = await Interpreter.fromAsset(
        'assets/y_eye_v2_adam.tflite',
        // 'assets/train_merged_mov2.tflite',//  별로
        // 'assets/train_merged_cnn_light.tflite',// 구림
        options: options,
      );
      // 모델 정보 출력
      print('Input tensor shape: ${_interpreter!.getInputTensor(0).shape}');
      print('Output tensor shape: ${_interpreter!.getOutputTensor(0).shape}');
      print('Input tensor type: ${_interpreter!.getInputTensor(0).type}');
      print('Output tensor type: ${_interpreter!.getOutputTensor(0).type}');
    } catch (e) {
      // ignore: avoid_print
      print('Error loading model: $e');
    }
  }

  // EAR 계산 함수
  double calculateEAR(List<double> keypoints) {
    try {
      // 키포인트를 (x,y) 쌍으로 변환
      List<Point> points = [];
      for (int i = 0; i < keypoints.length; i += 2) {
        points.add(Point(keypoints[i], keypoints[i + 1]));
        //print('Point ${i ~/ 2}: (${keypoints[i]}, ${keypoints[i + 1]})');
      }

      // 수직 거리 계산 (Python의 np.abs() 대신 dart:math의 abs() 사용)
      num verticalDist1 = (points[1].y - points[5].y).abs(); // 첫 번째 수직선
      num verticalDist2 = (points[2].y - points[4].y).abs(); // 두 번째 수직선

      // 수평 거리 계산
      num horizontalDist = (points[0].x - points[3].x).abs(); // 양 끝점 사이의 거리

      // 디버깅을 위한 거리값 출력
      print('Vertical1: $verticalDist1');
      print('Vertical2: $verticalDist2');
      print('Horizontal: $horizontalDist');

      // 0으로 나누기 방지
      if (horizontalDist < 0.000001) {
        print('Warning: Horizontal distance is too small');
        return 0.0;
      }

      // EAR 계산
      double ear = (verticalDist1 + verticalDist2) / (2.0 * horizontalDist);
      print('Calculated EAR: $ear');

      return ear;
    } catch (e) {
      print('Error in EAR calculation: $e');
      return 0.0;
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

          // MobileNetV2 전처리에 맞춰 -1에서 1 사이로 정규화
          _inputBuffer![y][x][0] = (r / 127.5) - 1.0;
          _inputBuffer![y][x][1] = (g / 127.5) - 1.0;
          _inputBuffer![y][x][2] = (b / 127.5) - 1.0;
        }
      }

      // 모델 실행을 위한 입력 준비
      final input = [_inputBuffer!];
      // softmax 출력을 위한 shape 수정 [1, 2]
      final output = [_outputBuffer];

      // 모델 실행
      _interpreter!.run(input, output);

      // 출력값을 _outputBuffer에 올바르게 복사
      for (int i = 0; i < _outputBuffer.length; i++) {
        _outputBuffer[i] = output[0][i];
      }

      // EAR 계산
      double ear = calculateEAR(_outputBuffer);
      _updateEyeState(ear); // 상태 업데이트
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
              child: Column(
                children: [
                  Text(
                    '현재 EAR: ${calculateEAR(_outputBuffer).toStringAsFixed(3)}',
                    style: const TextStyle(color: Colors.white),
                  ),
                  Text(
                    '상태: $eyeState',
                    style: TextStyle(
                        color: stateColor, fontWeight: FontWeight.bold),
                  ),
                  if (drowsyFrameCount > 0)
                    Text(
                      '연속 감지: $drowsyFrameCount',
                      style: const TextStyle(color: Colors.yellow),
                    ),
                ],
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
            Text('현재 EAR: ${calculateEAR(_outputBuffer).toStringAsFixed(3)}'),
            const Text('임계값: $earThreshold'),
            Text('눈 상태: $eyeState'),
            Text('연속 프레임: $drowsyFrameCount / $drowsyFrameThreshold'),
            const Text('알림 간격: 5초'),
            if (lastAlertTime != null)
              Text(
                  '마지막 알림: ${lastAlertTime!.toLocal().toString().split('.')[0]}'),
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
