import io
import cv2
import sys
import base64
import requests
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def breaker(num: int=50, char: str="*") -> None:
    print("\n" + num*char + "\n")


def decode_image(imageData) -> np.ndarray:
    header, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    if len(image.shape) == 4:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    return header, image


def show_image(image: np.ndarray, cmap: str="gnuplot2", title: str=None) -> None:
    plt.figure()
    plt.imshow(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB), cmap=cmap)
    plt.axis("off")
    if title: plt.title(title)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


def main():
    args_1: str = "--base-url"
    args_2: str = "--mode"

    base_url: str = "http://127.0.0.1:4004"
    mode: str = "/"

    if args_1 in sys.argv: base_url = sys.argv[sys.argv.index(args_1)+1]
    if args_2 in sys.argv: mode = sys.argv[sys.argv.index(args_2)+1]
    
    breaker()
    if mode == "/":
        url: str = base_url + mode
        response = requests.request(method="GET", url=url)
        if response.status_code == 200:
            print(f"Status Text: {response.json()['statusText']}")
        else:
            print("Error: " + str(response.status_code))

    elif mode == "version":
        url: str = base_url + "/" + mode
        response = requests.request(method="GET", url=url)
        if response.status_code == 200:
            print(f"Version: {response.json()['version']}")
        else:
            print("Error: " + str(response.status_code))
        
    elif mode == "distribution":
        feature_name: str = sys.argv[sys.argv.index(args_2) + 2]

        url: str = base_url + "/" + mode + "/" + f"{feature_name}"

        response = requests.request(method="GET", url=url)
        if response.status_code == 200:
            _, image = decode_image(imageData=response.json()["imageData"])
            print(f"Status Text: {response.json()['statusText']}")
            show_image(image=image, title=response.json()["message"])
        else:
            print("Error: " + str(response.status_code))

    elif mode == "train":
        url: str = base_url + "/" + mode

        response = requests.request(method="GET", url=url)
        if response.status_code == 200:
            print(f"Status Text: {response.json()['statusText']}")
            breaker()
            print(f"Best ACC Model : {response.json()['best_acc_model']}")
        else:
            print("Error: " + str(response.status_code))
    
    elif mode == "logs":
        model_name: str = sys.argv[sys.argv.index(args_2) + 2]
        fold: str = sys.argv[sys.argv.index(args_2) + 3]

        url: str = base_url + "/" + "train/logs" + "/" + model_name + "/" + fold

        response = requests.request(method="GET", url=url)
        if response.status_code == 200:
            print(f"Status Text: {response.json()['statusText']}")
            breaker()
            logs: dict = response.json()["logs"]
            for key, value in logs.items():
                if key != "name" :
                    if key == "fold":
                        print(f"{key.title():>11}: {int(value)}")
                    else:
                        print(f"{key.title():>11}: {float(value):.5f}")
                else:
                    print(f"{key.title():>11}: {value}")
        else:
            print("Error: " + str(response.status_code))
    
    elif mode == "infer":
        url: str = base_url + "/" + mode

        # 0.36000
        payload: dict = {
            "solidity" : 16,
            "eccentricity" : 0.36000, 
            "equiv_diameter" : 9,
            "extrema" : 4,
            "filled_area" : 21,
            "extent" : 3.5,
            "orientation" : 8, 
            "euler_number" : 22.9,
            "major_axis_length" : 4.5,
            "minor_axis_length" : 1.5,
            "perimeter" : 15,
            "convex_area" : 5,
            "area" : 21,
            "raddi" : 0.9,
        }

        response = requests.request(method="POST", url=url, json=payload)
        if response.status_code == 200:
            print(f"Probability: {float(response.json()['probability']):.5f}")
        else:
            print("Error: " + str(response.status_code))
    
    breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)