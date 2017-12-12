from digicampipe.image.sky_image import *
from digicampipe.image.cones_image import ConesImage
import os


def main():
    # find stars in lid CCD images:
    images_directory = './data/test/star_on_lid'
    output_directory = './data/test/star_on_lid'
    image_files = []
    for file in os.listdir(images_directory):
        if file.endswith(".fits"):
            image_files.append(os.path.join(images_directory, file))
    crop_pixels1 = [#(850, 50),
                    #(550, 500),
                    (350, 900),
                    #(600, 1350),
                    (850, 1800),]
    crop_pixels2 = [#(1100, 650),
                    #(900, 1000),
                    (770, 1550),
                    #(1000, 2050),
                    (1300, 2400),]
    """
    crop_pixels1 = []
    crop_pixels2 = []
    npixelx=3201
    stepx=400
    npixely=2401
    stepy=300
    for x1, x2 in zip(range(0, 8*stepx+1, stepx), range(stepx, npixelx, stepx)):
        for y1, y2 in zip(range(0, 8*stepy, stepy), range(stepy, npixely, stepy)):
            crop_pixels1.append((x1, y1))
            crop_pixels2.append((x2, y2))
    """
#    lidccd_obs = LidCCDObservation(image_files, crop_pixels1, crop_pixels2,
#                                   scale_low_images_deg=8., scale_high_images_deg=12.)
#    lidccd_obs.plot_image_solved(output_directory)
#    lidccd_obs.plot_image_treated(output_directory)
#    lidccd_obs.print_summary()

    # find cones in lid CCD images:

#    cones_img=ConesImage('./data/test/cones_1509411741.fits', './data/test/cones_1509411741-cone.fits')
    cones_img = ConesImage('test', './data/test/cone_simu/cone.fits')

#    cones_img.plot_cones(output_dir='./data/test/cone_simu')
#    cones_img.plot_fft_cones(output_dir='./data/test/cone_simu')
#    cones_img.get_cones_separation_reciprocal(output_dir='./data/test/cone_simu')
#    cones_img.plot_fft_cones(output_dir='./data/test/cone_simu', radius_mask=2.1)
#    cones_img.scan_cone_position(output_dir='./data/test/cone_simu', radius_mask=2.1,
#                                 center_scan=cones_img.center_fitted, rotations=(60,120,180,240,300))
#    cones_img.get_cone(output_dir='./data/test/cone_simu', radius_mask=2.1)
#    cones_img.plot_cones(output_dir='./data/test/cone_simu', radius_mask=2.1)
    cones_img.plot_cones_presence(output_dir='./data/test/cone_simu', radius_mask=15.5)
    """
    mask, ks = cones_img.get_fft_mask(radius=2.1)
    image_cones = np.real(np.fft.ifft2(cones_img.fft_image_cones * mask))
    center_image = (np.array(image_cones.shape[::-1]) - 1) / 2
    real_offset = np.array((55, 44))
    r1 = 58 * np.array((0.5, np.sqrt(3) / 2))
    r2 = 58 * np.array((0.5, -np.sqrt(3) / 2))
    for offset_from_real in (0, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2):
        pos = center_image + real_offset + offset_from_real
        hex = get_neg_hexagonalicity_with_mask(pos, image_cones, r1, r2)
        print("offset_from_real=", offset_from_real,'hex=', hex)
    """
#    cones_img.plot_cone(output_dir='./data/test', radius_mask=2.1)
#    cones_img.scan_cone_position(output_dir='./data/test', radius_mask=2.1)
    print("done")

if __name__ == '__main__':
    main()
