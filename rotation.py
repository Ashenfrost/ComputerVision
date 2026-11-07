import cv2
import numpy as np
import os
import random
import glob


def roter_billede(billede, vinkel):
    """
    Roterer et billede med en given vinkel (i grader) uden at beskære det.
    """
    h, w = billede.shape[:2]
    center = (w // 2, h // 2)

    # Rotationsmatrix
    rotation_matrix = cv2.getRotationMatrix2D(center, vinkel, 1.0)

    # Beregn ny billedstørrelse så hele det roterede billede er med
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    ny_w = int((h * sin) + (w * cos))
    ny_h = int((h * cos) + (w * sin))

    # Juster rotationsmatricen til den nye størrelse
    rotation_matrix[0, 2] += (ny_w / 2) - center[0]
    rotation_matrix[1, 2] += (ny_h / 2) - center[1]

    roteret = cv2.warpAffine(billede, rotation_matrix, (ny_w, ny_h))
    return roteret


def roter_billeder_tilfaeldigt(input_mappe, output_mappe, min_vinkel=-180, max_vinkel=180, antal_pr_billede=1):
    """
    Går igennem alle billeder i input_mappe, roterer dem tilfældigt
    og gemmer resultaterne i output_mappe.

    Args:
        input_mappe: Sti til mappen med originale billeder.
        output_mappe: Sti hvor de roterede billeder skal gemmes.
        min_vinkel: Mindste rotationsvinkel (grader).
        max_vinkel: Største rotationsvinkel (grader).
        antal_pr_billede: Hvor mange tilfældigt roterede versioner der laves pr. billede.
    """
    os.makedirs(output_mappe, exist_ok=True)

    # Understøttede filtyper
    filtyper = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    billedstier = []
    for ft in filtyper:
        billedstier.extend(glob.glob(os.path.join(input_mappe, ft)))

    if not billedstier:
        print(f"Ingen billeder fundet i {input_mappe}")
        return

    for sti in billedstier:
        billede = cv2.imread(sti)

        if billede is None:
            print(f"Kunne ikke læse billedet: {sti}")
            continue

        filnavn = os.path.splitext(os.path.basename(sti))[0]
        endelse = os.path.splitext(sti)[1]

        for i in range(antal_pr_billede):
            vinkel = random.uniform(min_vinkel, max_vinkel)
            roteret = roter_billede(billede, vinkel)

            nyt_filnavn = f"{filnavn}_rotated_{i}_{int(vinkel)}deg{endelse}"
            output_sti = os.path.join(output_mappe, nyt_filnavn)

            cv2.imwrite(output_sti, roteret)
            print(f"Gemt: {output_sti} (vinkel: {vinkel:.1f}°)")


if __name__ == "__main__":
    input_mappe = "Original"           # Mappe med dine originale billeder
    output_mappe = "LE ROTATIONES" # Mappe hvor resultaterne gemmes

    roter_billeder_tilfaeldigt(
        input_mappe=input_mappe,
        output_mappe=output_mappe,
        min_vinkel=-45,
        max_vinkel=45,
        antal_pr_billede=3
    )