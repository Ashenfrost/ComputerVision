import cv2
import numpy as np
import os
import glob



def keep_hues_in_range(billede, min_hue, max_hue):
    """
    Beholder kun pixels hvis hue ligger indenfor [min_hue, max_hue].

    Args:
        billede: Inputbillede i BGR-format.
        min_hue: Mindste farvetoneværdi.
        max_hue: Største farvetoneværdi.

    Returns:
        Billedet hvor pixels udenfor hue-intervallet er sat til sort.
    """
    hsv = cv2.cvtColor(billede, cv2.COLOR_BGR2HSV)
    min_h = int(min_hue)
    max_h = int(max_hue)
    if max_h > 179:
        max_h = 179

    lower = np.array([min_h, 0, 0], dtype=np.uint8)
    upper = np.array([max_h, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(billede, billede, mask=mask)
    return result

def rehue_billeder(input_mappe, output_mappe, min_hue=1, max_hue=15):
    """
    Går igennem alle billeder i input_mappe, fjerner hues udenfor intervallet
    [min_hue, max_hue] og gemmer resultatet i output_mappe.

    Args:
        input_mappe: Sti til mappen med originale billeder.
        output_mappe: Sti hvor de ændrede billeder skal gemmes.
        min_hue: Mindste farvetoneværdi.
        max_hue: Største farvetoneværdi.
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

        filtered = keep_hues_in_range(billede, min_hue, max_hue)
        nyt_filnavn = f"{filnavn}_huefiltered_{int(min_hue)}-{int(max_hue)}{endelse}"
        output_sti = os.path.join(output_mappe, nyt_filnavn)
        cv2.imwrite(output_sti, filtered)
        print(f"Gemt: {output_sti} (beholder hues: {min_hue}-{max_hue})")


if __name__ == "__main__":  
    input_mappe = "strawberry"           # Mappe med dine originale billeder
    output_mappe = "REDDEST" # Mappe hvor resultaterne gemmes

    rehue_billeder(
        input_mappe=input_mappe,
        output_mappe=output_mappe,
        min_hue=1,
        max_hue=15,
    )