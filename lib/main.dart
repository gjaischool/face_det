import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.dark(),
      home: BrightnessDetection(cameras: cameras),
    );
  }
}

class BrightnessDetection extends StatefulWidget {
  final List<CameraDescription> cameras;

  const BrightnessDetection({super.key, required this.cameras});

  @override
  _BrightnessDetectionState createState() => _BrightnessDetectionState();
}

class _BrightnessDetectionState extends State<BrightnessDetection> {
  late CameraController _controller;
  bool isProcessing = false;
  final audioPlayer = AudioPlayer();

  // 어두움 감지를 위한 상태 변수들
  int darkFrameCount = 0;
  static const int DARK_FRAME_THRESHOLD = 15; //15프레임 연속 감지되면 알람
  static const int BRIGHTNESS_THRESHOLD = 60; //이 값 이해로 내려가면 어두움
  DateTime? lastAlertTime;
  double currentBrightness = 0.0;
  bool isAlarmPlaying = false; // 알람 재생 상태 추적

  @override
  void initState() {
    super.initState();
    _initializeCamera();

    // 알람 완료 이벤트 리스너
    audioPlayer.onPlayerComplete.listen((event) {
      if (isAlarmPlaying) {
        _startAlarm(); // 여전히 어두운 상태면 알람 계속 재생
      }
    });
  }

  Future<void> _initializeCamera() async {
    _controller = CameraController(
      widget.cameras[1], // 전면 카메라
      ResolutionPreset.medium,
      enableAudio: false,
    );

    try {
      await _controller.initialize();

      _controller.startImageStream((CameraImage image) {
        if (!isProcessing) {
          isProcessing = true;
          _processImage(image);
        }
      });

      setState(() {});
    } catch (e) {
      print('카메라 초기화 오류: $e');
    }
  }

  void _processImage(CameraImage image) {
    try {
      final bytes = image.planes[0].bytes;
      int sum = 0;
      for (int i = 0; i < bytes.length; i++) {
        sum += bytes[i];
      }
      currentBrightness = sum / bytes.length;

      setState(() {
        if (currentBrightness < BRIGHTNESS_THRESHOLD) {
          darkFrameCount++;
          if (darkFrameCount >= DARK_FRAME_THRESHOLD && !isAlarmPlaying) {
            _startAlarm();
          }
        } else {
          // 밝기가 정상으로 돌아오면
          if (darkFrameCount >= DARK_FRAME_THRESHOLD) {
            _stopAlarm(); // 알람 중지
          }
          darkFrameCount = 0;
        }
      });
    } finally {
      isProcessing = false;
    }
  }

  void _startAlarm() {
    final now = DateTime.now();
    if (lastAlertTime == null ||
        now.difference(lastAlertTime!) > const Duration(seconds: 1)) {
      isAlarmPlaying = true;
      audioPlayer.play(AssetSource('alert.wav'));
      lastAlertTime = now;

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              '어두움이 감지되었습니다!',
              style: TextStyle(fontSize: 16),
            ),
            backgroundColor: Colors.red,
            duration: Duration(seconds: 2),
          ),
        );
      }
    }
  }

  void _stopAlarm() {
    if (isAlarmPlaying) {
      Future.delayed(const Duration(seconds: 3), () {
        if (isAlarmPlaying) {
          // 알람이 아직 재생 중인 경우에만 중지
          isAlarmPlaying = false;
          audioPlayer.stop();

          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text(
                  '정상 상태로 돌아왔습니다.',
                  style: TextStyle(fontSize: 16),
                ),
                backgroundColor: Colors.green,
                duration: Duration(seconds: 2),
              ),
            );
          }
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('졸음 감지 테스트'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _showSettingsDialog,
          ),
        ],
      ),
      body: Stack(
        children: [
          CameraPreview(_controller),
          Positioned(
            top: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '현재 밝기: ${currentBrightness.toStringAsFixed(1)}',
                    style: const TextStyle(color: Colors.white, fontSize: 16),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '어두운 프레임: $darkFrameCount',
                    style: const TextStyle(color: Colors.white, fontSize: 16),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '알람 상태: ${isAlarmPlaying ? "켜짐" : "꺼짐"}',
                    style: TextStyle(
                      color: isAlarmPlaying ? Colors.red : Colors.green,
                      fontSize: 16,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _showSettingsDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('상태 정보'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('밝기 임계값: $BRIGHTNESS_THRESHOLD'),
            const Text('감지 프레임 수: $DARK_FRAME_THRESHOLD'),
            Text('알람 상태: ${isAlarmPlaying ? "켜짐" : "꺼짐"}'),
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
    audioPlayer.dispose();
    super.dispose();
  }
}
