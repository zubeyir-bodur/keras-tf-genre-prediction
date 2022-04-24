import os
import glob


def main():
    """
    Renames the incoming files with zeros in the beginning
    removed, so that movie ID's match with the filenames exactly
    :return:
    """
    ratios = [30, 37, 44, 51, 58, 100]
    for r in ratios:
        file_names = glob.glob("../img/" + str(r) + "/*")
        for file_name in file_names:
            base_name = os.path.basename(file_name)
            new_name = base_name
            for char in base_name:
                if char == '0':
                    new_name = new_name[1:]
                else:
                    break
            print(new_name)
            os.rename(file_name, "../img/" + str(r) + "/" + new_name)


if __name__ == "__main__":
    main()
