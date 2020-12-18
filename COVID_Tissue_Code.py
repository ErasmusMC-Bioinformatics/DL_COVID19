""" RGB color code dictionary for NSCLC growth patterns and adjacent tissue for six-class classification """
""" Furthermore,  RGB color code dictionary for individual segmentations of the objects of interest in three-class classification """

color_dict = {0: [255, 255, 255],  # white - background
              1: [0, 255, 0],  # green lime- other
              2: [0, 0, 255],  # blue- blood vessel
              3: [255, 0, 0],  # red- hemorrhage
              4: [255, 255, 0],  # yellow- bronchus
              5: [255, 0, 255] #magenta- thrombus
              }

color_dict_hemorrhage = {0: [255, 255, 255],  # white - background
              1: [0, 255, 0],  # green lime- other
              2: [255, 0, 0],  # red- hemorrhage
              }

color_dict_thrombi = {0: [255, 255, 255],  # white - background
              1: [0, 255, 0],  # green lime- other
              2: [255, 0, 255],  # magenta- thrombus
              }

color_dict_vessels = {0: [255, 255, 255],  # white - background
              1: [0, 255, 0],  # green lime- other
              2: [0, 0, 255],  # blue -blood vessels
              }

color_dict_bronchus = {0: [255, 255, 255],  # white - background
              1: [0, 255, 0],  # green lime- other
              2: [255, 255, 0],  # yellow- bronchus
              }