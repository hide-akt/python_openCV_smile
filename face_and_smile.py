import sys # インタプリタや実行環境に関する情報を扱うためのライブラリ
import cv2 # OpenCVのインポート

# VideoCaptureのインスタンスを作成する
# 引数でカメラを選択

cap = cv2.VideoCapture(1)

if cap.isOpened() is False:
    print("can not open camera")
    sys.exit()

# カスケード評価器を埋め込み
# https://github.com/opencv/opencv/tree/master/data/haarcascades

cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    # 処理速度を考慮しリサイズ
    frame = cv2.resize(frame, (int(frame.shape[1]*0.7), int(frame.shape[0]*0.7)))

    #処理速度高めるため画像をグレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出
    facerect = cascade.detectMultiScale(
        gray,
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )

    if len(facerect) != 0:
        for x, y, w, h in facerect:
            # 顔の部分(この顔の部分に対して目の検出をかける)
            smile_gray = gray[y: y + h, x: x + w]

            # 顔の部分から笑顔を検出
            smile = smile_cascade.detectMultiScale(
                smile_gray,
                scaleFactor=1.11, #PCスペックに依存するので適宜修正1.11
                minNeighbors=3,
                minSize=(100, 100)
            )

            if len(smile) == 1:
                cv2.putText(
                    frame,
                    "Nice smile!",
                    (425,270), #位置を修正
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            else:
                cv2.putText(
                    frame,
                    "No smile...",
                    (425,270), #位置を修正
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # 顔検出した部分に枠を描画
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                thickness=2
                )
    cv2.imshow('frame', frame)

    # キー入力を1ms待って、kが27(ESC)だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

# キャプチャをリリースして、ウィンドウを全て閉じる
cap.release()
cv2.destroyAllWindows