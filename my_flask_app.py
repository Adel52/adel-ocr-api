import flask
from flask import request, jsonify
import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Input
import io # لاستقبال الصورة من الطلب

# --- إعدادات عامة ---
TARGET_CLASS_NAME_YOLO = 'national_id'
CONFIDENCE_THRESHOLD_YOLO = 0.25
# مجلد مؤقت على PythonAnywhere (يمكنك استخدام /tmp أو مجلد داخل مشروعك)
TEMP_DIGIT_SEGMENTS_FOLDER = '/tmp/temp_ocr_digit_segments' # أو './temp_ocr_digit_segments' إذا كان داخل مجلد المشروع

# --- مسارات ملفات الأوزان (يجب تعديلها بعد رفعها إلى PythonAnywhere) ---
# مثال: إذا وضعت ملفات الأوزان في مجلد "models" داخل مشروعك على PythonAnywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # مسار المجلد الحالي للكود
MODEL_PATH_YOLO_LAST = os.path.join(BASE_DIR, 'models', 'last.pt') # سنعتمد على هذا فقط لـ YOLO
MODEL_WEIGHTS_PATH_CNN = os.path.join(BASE_DIR, 'models', 'ar_numbers_v6.h5')

# --- تحميل النماذج مرة واحدة عند بدء تشغيل التطبيق ---
yolo_model = None
cnn_model = None

def load_models():
    global yolo_model, cnn_model
    print("بدء تحميل النماذج...")

    # --- تحميل موديل YOLO ---
    print(f"محاولة تحميل موديل YOLO من: {MODEL_PATH_YOLO_LAST}")
    try:
        yolo_model = YOLO(MODEL_PATH_YOLO_LAST)
        print(f"تم تحميل موديل YOLOv8 بنجاح من: {MODEL_PATH_YOLO_LAST}")
    except Exception as e_yolo:
        print(f"فشل تحميل موديل YOLO من {MODEL_PATH_YOLO_LAST}: {e_yolo}")
        yolo_model = None

    # --- تحميل موديل CNN ---
    if yolo_model: # لن نحاول تحميل CNN إذا فشل تحميل YOLO لتوفير الموارد
        print(f"محاولة تحميل موديل CNN للأرقام من: {MODEL_WEIGHTS_PATH_CNN}")
        try:
            input_shape_val = 64
            no_classes_val = 10
            cnn_model = Sequential(name="Arabic_Digits_CNN")
            cnn_model.add(Input(shape=(input_shape_val, input_shape_val, 1), name='Input_Layer'))
            cnn_model.add(Conv2D(32, (3,3), strides=1, activation='relu', name='Conv2D_1'))
            cnn_model.add(MaxPooling2D(pool_size=(2,2), name='MaxPooling2D_1'))
            cnn_model.add(Conv2D(64, (3,3), activation='relu', name='Conv2D_2'))
            cnn_model.add(MaxPooling2D(pool_size=(2,2), name='MaxPooling2D_2'))
            cnn_model.add(Conv2D(128, (3,3), activation='relu', name='Conv2D_3'))
            cnn_model.add(MaxPooling2D(pool_size=(2,2), name='MaxPooling2D_3'))
            cnn_model.add(Flatten(name='Flatten_Layer'))
            cnn_model.add(Dense(units=500, activation='relu', name='Dense_1'))
            cnn_model.add(BatchNormalization(name='BatchNormalization_1'))
            cnn_model.add(Dropout(0.4, name='Dropout_1'))
            cnn_model.add(Dense(units=250, activation='relu', name='Dense_2'))
            cnn_model.add(BatchNormalization(name='BatchNormalization_2'))
            cnn_model.add(Dropout(0.4, name='Dropout_2'))
            cnn_model.add(Dense(units=100, activation='relu', name='Dense_3'))
            cnn_model.add(BatchNormalization(name='BatchNormalization_3'))
            cnn_model.add(Dropout(0.4, name='Dropout_3'))
            cnn_model.add(Dense(units=no_classes_val, activation='softmax', name='Output_Layer'))
            cnn_model.load_weights(MODEL_WEIGHTS_PATH_CNN)
            print(f"تم تحميل أوزان نموذج CNN للأرقام بنجاح من: {MODEL_WEIGHTS_PATH_CNN}")
        except Exception as e_cnn:
            print(f"حدث خطأ أثناء بناء أو تحميل نموذج CNN للأرقام: {e_cnn}")
            cnn_model = None
    else:
        print("تم تخطي تحميل موديل CNN بسبب فشل تحميل موديل YOLO.")

    print("انتهى تحميل النماذج.")

# استدعاء تحميل النماذج مرة واحدة
load_models()

# --- الدوال المساعدة (كما هي مع تعديلات طفيفة) ---
def extract_national_id_roi_yolo(full_id_image_bgr, yolo_detector):
    if yolo_detector is None:
        print("   موديل YOLO غير محمل، لا يمكن استخلاص منطقة الرقم القومي.")
        return None
    print("   جاري اكتشاف منطقة الرقم القومي بواسطة YOLO...")
    results = yolo_detector(full_id_image_bgr, verbose=False, conf=CONFIDENCE_THRESHOLD_YOLO)
    national_id_roi_extracted_bgr = None
    highest_confidence_yolo = 0.0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = yolo_detector.names[class_id]
            if class_name == TARGET_CLASS_NAME_YOLO and confidence > highest_confidence_yolo:
                highest_confidence_yolo = confidence
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                ymin, ymax = max(0, xyxy[1]), min(full_id_image_bgr.shape[0], xyxy[3])
                xmin, xmax = max(0, xyxy[0]), min(full_id_image_bgr.shape[1], xyxy[2])
                if ymin < ymax and xmin < xmax:
                    national_id_roi_extracted_bgr = full_id_image_bgr[ymin:ymax, xmin:xmax]
    if national_id_roi_extracted_bgr is not None:
        print(f"   تم اكتشاف منطقة '{TARGET_CLASS_NAME_YOLO}' بواسطة YOLO بثقة {highest_confidence_yolo:.2f}%.")
    else:
        print(f"   لم يتم اكتشاف منطقة '{TARGET_CLASS_NAME_YOLO}' بواسطة YOLO بالثقة المطلوبة.")
    return national_id_roi_extracted_bgr

def segment_digits_from_roi_user_code(nid_roi_image_bgr, output_folder=TEMP_DIGIT_SEGMENTS_FOLDER,
                                     canvas_size=(64, 64), digit_resize_intermediate=(28, 48)):
    print(f"   جاري تقسيم الأرقام من منطقة الرقم القومي إلى مجلد: {output_folder} باستخدام منطق المستخدم...")
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    img_gray_roi = cv2.cvtColor(nid_roi_image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 10 < w < 100 and 30 < h < 100:
            boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: b[0], reverse=True)
    num_detected_boxes = len(boxes)
    if num_detected_boxes != 14 :
        print(f"   ⚠️ تحذير (منطق المستخدم): تم اكتشاف {num_detected_boxes} رقمًا مرشحًا بدلاً من 14.")
        if num_detected_boxes > 14:
            boxes = boxes[:14]
        elif num_detected_boxes == 0:
            print("   لم يتم العثور على أي أرقام مرشحة للتقسيم.")
            return []
    saved_digit_paths = []
    for i, (x, y, w, h) in enumerate(boxes):
        digit_cropped_from_roi = img_gray_roi[y:y+h, x:x+w]
        if digit_cropped_from_roi.size == 0: continue
        digit_resized_intermediate = cv2.resize(digit_cropped_from_roi, digit_resize_intermediate, interpolation=cv2.INTER_AREA)
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255
        x_offset = (canvas_size[0] - digit_resize_intermediate[0]) // 2
        y_offset = (canvas_size[1] - digit_resize_intermediate[1]) // 2
        canvas[y_offset:y_offset+digit_resize_intermediate[1], x_offset:x_offset+digit_resize_intermediate[0]] = digit_resized_intermediate
        filename = os.path.join(output_folder, f'digit_{i+1:02d}.png')
        cv2.imwrite(filename, canvas)
        saved_digit_paths.append(filename)
    if saved_digit_paths:
        print(f"   ✅ تم حفظ {len(saved_digit_paths)} رقمًا مقسمًا (منطق المستخدم) في: {os.path.abspath(output_folder)}")
    else:
        print("   لم يتم حفظ أي أرقام مقسمة.")
    return saved_digit_paths

def preprocess_cnn_input(digit_image_path, target_size=(64, 64)):
    try:
        img_gray = cv2.imread(digit_image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"خطأ في preprocess_cnn_input: لم يتمكن من قراءة الصورة {digit_image_path}")
            return None
        if img_gray.shape != target_size:
            img_gray = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)
        img_normalized = img_gray / 255.0
        img_reshaped = img_normalized.reshape(1, target_size[0], target_size[1], 1)
        return img_reshaped
    except Exception as e:
        print(f"خطأ في preprocess_cnn_input for {digit_image_path}: {e}")
        return None

# --- Flask App ---
app = flask.Flask(__name__) # اسم التطبيق مهم للـ WSGI

@app.route('/verify_national_id', methods=['POST'])
def verify_national_id():
    global yolo_model, cnn_model # للتأكد من استخدام النماذج المحملة

    if yolo_model is None: # نتحقق من YOLO أولاً
        return jsonify({
            "status": "error",
            "message": "موديل YOLO غير محمل بشكل صحيح على الخادم.",
            "extracted_id": None,
            "match": False
        }), 500
    if cnn_model is None: # ثم نتحقق من CNN
        return jsonify({
            "status": "error",
            "message": "موديل CNN للأرقام غير محمل بشكل صحيح على الخادم.",
            "extracted_id": None,
            "match": False
        }), 500


    if 'image' not in request.files:
        return jsonify({
            "status": "error",
            "message": "لم يتم إرسال ملف الصورة.",
            "extracted_id": None,
            "match": False
        }), 400

    if 'national_id_typed' not in request.form:
        return jsonify({
            "status": "error",
            "message": "لم يتم إرسال الرقم القومي المدخل.",
            "extracted_id": None,
            "match": False
        }), 400

    image_file = request.files['image']
    national_id_typed_by_user = request.form['national_id_typed']

    # تنظيف المجلد المؤقت قبل البدء (إذا كان موجودًا من عملية سابقة فاشلة)
    if os.path.exists(TEMP_DIGIT_SEGMENTS_FOLDER):
        try:
            shutil.rmtree(TEMP_DIGIT_SEGMENTS_FOLDER)
            print(f"تم تنظيف المجلد المؤقت القديم: {TEMP_DIGIT_SEGMENTS_FOLDER}")
        except Exception as e_clean_init:
            print(f"خطأ بسيط أثناء تنظيف المجلد المؤقت القديم: {e_clean_init}")


    try:
        # قراءة الصورة من الطلب
        in_memory_file = io.BytesIO()
        image_file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        original_id_image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if original_id_image_bgr is None:
            return jsonify({
                "status": "error",
                "message": "ملف الصورة غير صالح أو لا يمكن قراءته.",
                "extracted_id": None,
                "match": False
            }), 400

        nid_roi_bgr = extract_national_id_roi_yolo(original_id_image_bgr, yolo_model)

        final_national_id_extracted_str = None # نهيئ المتغير

        if nid_roi_bgr is not None:
            segmented_digit_image_paths = segment_digits_from_roi_user_code(
                nid_roi_bgr,
                output_folder=TEMP_DIGIT_SEGMENTS_FOLDER,
                canvas_size=(64,64),
                digit_resize_intermediate=(28,48)
            )

            if segmented_digit_image_paths and len(segmented_digit_image_paths) > 0:
                print(f"   جاري التعرف على {len(segmented_digit_image_paths)} رقمًا مقسمًا بواسطة CNN...")
                final_recognized_id_list = []

                for digit_path in segmented_digit_image_paths:
                    processed_cnn_input = preprocess_cnn_input(digit_path)
                    if processed_cnn_input is not None:
                        prediction = cnn_model.predict(processed_cnn_input, verbose=0)
                        recognized_digit = np.argmax(prediction[0])
                        final_recognized_id_list.append(str(recognized_digit))
                    else:
                        final_recognized_id_list.append("?") # علامة خطأ

                final_national_id_extracted_str = "".join(reversed(final_recognized_id_list))
                print("-" * 40)
                print(f"الرقم القومي النهائي المستخرج هو: {final_national_id_extracted_str}")
                print("-" * 40)

                # التحقق من صحة الرقم المستخرج (بسيط)
                is_valid_format = True
                clean_id_for_check = final_national_id_extracted_str.replace('?','')
                if len(clean_id_for_check) != 14 or not clean_id_for_check.isdigit():
                    is_valid_format = False
                    print(f"تحذير: الرقم القومي المستخرج ({final_national_id_extracted_str}) قد لا يكون صحيحًا.")

                # المقارنة
                match_status = (final_national_id_extracted_str == national_id_typed_by_user) and is_valid_format

                return jsonify({
                    "status": "success",
                    "message": "تمت معالجة الصورة بنجاح.",
                    "extracted_id": final_national_id_extracted_str,
                    "typed_id": national_id_typed_by_user,
                    "match": match_status,
                    "is_extracted_id_valid_format": is_valid_format
                }), 200
            else:
                message = "لم يتم تقسيم أي أرقام بنجاح من منطقة الرقم القومي."
        else:
            message = "لم يتمكن موديل YOLO من استخلاص منطقة الرقم القومي."

        # إذا وصلنا هنا، فهذا يعني أن العملية لم تكتمل بنجاح كامل
        # ولكن لم يحدث خطأ فادح يوقف البرنامج
        return jsonify({
            "status": "processing_issue", # نوع حالة جديد للإشارة لمشكلة في المعالجة وليس خطأ فني
            "message": message,
            "extracted_id": final_national_id_extracted_str, # قد يكون None أو قيمة جزئية
            "typed_id": national_id_typed_by_user,
            "match": False
        }), 200 # نرجع 200 لأن الطلب عولج لكن النتيجة ليست مثالية


    except Exception as e_main:
        print(f"حدث خطأ رئيسي في معالجة الطلب: {e_main}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"خطأ داخلي في الخادم: {str(e_main)}",
            "extracted_id": None,
            "match": False
        }), 500
    finally:
        # تنظيف المجلد المؤقت في كل الأحوال
        if os.path.exists(TEMP_DIGIT_SEGMENTS_FOLDER):
            try:
                shutil.rmtree(TEMP_DIGIT_SEGMENTS_FOLDER)
                print(f"تم حذف المجلد المؤقت: {TEMP_DIGIT_SEGMENTS_FOLDER}")
            except Exception as e_clean:
                print(f"خطأ أثناء حذف المجلد المؤقت: {e_clean}")

# هذا الجزء خاص بالتشغيل المحلي للاختبار فقط
# على PythonAnywhere، سيتم تشغيل التطبيق بواسطة خادم WSGI الخاص بهم
if __name__ == "__main__":
    # تأكد من وجود مجلد models وملفات الأوزان بداخله إذا كنت تشغل محليًا
    # أو عدل المسارات أعلاه لتناسب مكان تخزينك للملفات
    if not os.path.exists(MODEL_PATH_YOLO_LAST) or not os.path.exists(MODEL_WEIGHTS_PATH_CNN):
         print("تحذير: لم يتم العثور على ملفات النماذج في المسارات المحددة. قد لا يعمل التطبيق بشكل صحيح.")
         print(f"YOLO Last: {MODEL_PATH_YOLO_LAST} (Exists: {os.path.exists(MODEL_PATH_YOLO_LAST)})")
         print(f"CNN Weights: {MODEL_WEIGHTS_PATH_CNN} (Exists: {os.path.exists(MODEL_WEIGHTS_PATH_CNN)})")

    # اسم التطبيق هنا 'app' يجب أن يتطابق مع ما سيتم استيراده في ملف WSGI
    app.run(debug=True, host='0.0.0.0', port=5000) # للتشغيل المحلي