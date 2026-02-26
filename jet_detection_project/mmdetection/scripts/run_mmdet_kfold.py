import subprocess
import sys
from pathlib import Path

def main():
    # Bu script mmdetection/scripts altında olduğu için yollar aşağıdakı gibidir:
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_jet_detection.py"
    
    print("=" * 60)
    print("MMDetection Cascade R-CNN - 5-Fold Training Runner")
    print("=" * 60)

    if not train_script.exists():
        print(f"[ERROR] Eğitim scripti bulunamadı: {train_script}")
        return

    for fold in range(5):
        print(f"\n" + "-" * 50)
        print(f"[FOLD {fold} BAŞLIYOR...]")
        print("-" * 50)
        
        # Komutu hazırla
        cmd = [
            sys.executable,
            str(train_script),
            "--fold", str(fold)
        ]
        
        try:
            # Eğitimi başlat ve bitmesini bekle
            subprocess.run(cmd, check=True)
            print(f"\n[OK] Fold {fold} başarıyla tamamlandı.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Fold {fold} sırasında hata oluştu: {e}")
            print("İşlem durduruldu.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n[INFO] Kullanıcı tarafından durduruldu.")
            sys.exit(0)

    print("\n" + "=" * 60)
    print("[SUCCESS] Tüm 5 fold eğitimi başarıyla tamamlandı!")
    print("=" * 60)

if __name__ == "__main__":
    main()
