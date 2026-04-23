# 車両検査AIシステム

防犯カメラ画像から車両の損傷を自動検出するシステムです。

## 機能
- QRコードによる車両ID読み取り
- YOLOv8による損傷・異常検出（GPU対応）
- FastAPI による REST API 提供
- 検査結果のJSON出力

## 技術スタック
- Python / FastAPI
- YOLOv8（Ultralytics）
- OpenCV
- RTX 5060 Ti によるGPU推論

## 起動方法
pip install -r requirements.txt
python app.py

## API
- POST /inspect：画像をアップロードして検査
- GET /health：ヘルスチェック
