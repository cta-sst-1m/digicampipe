from digicampipe.image.cones_image import ConesImage


def get_cone_position_simu(output_dir=None):
    cones_img = ConesImage('test', output_dir=output_dir)
    cones_img.plot_cones(output_dir=output_dir)
    cones_img.plot_fft_cones(output_dir=output_dir)
    cones_img.get_cones_separation_reciprocal(output_dir=output_dir)
    cones_img.plot_fft_cones(output_dir=output_dir, radius_mask=2.1)
    # cones_img.scan_cone_position(output_dir=output_dir,
    #                              radius_mask=2.1,
    #                              center_scan=cones_img.center_fitted,
    #                              rotations=(60, 120, 180, 240, 300))
    cones_img.get_cone(output_dir=output_dir, radius_mask=2.1)
    cones_img.plot_cones(output_dir=output_dir, radius_mask=2.1)
    cones_img.plot_cones_presence(output_dir=output_dir,
                                  radius_mask=2.1)
    return cones_img.pixels_pos_predict, cones_img.pixels_pos_true


def get_cones_position(filename, output_dir=None):
    cones_img = ConesImage(filename)
    cones_img.plot_cones(output_dir=output_dir)
    cones_img.plot_fft_cones(output_dir=output_dir)
    cones_img.get_cones_separation_reciprocal(output_dir=output_dir)
    cones_img.plot_fft_cones(output_dir=output_dir, radius_mask=2.1)
    cones_img.scan_cone_position(output_dir=output_dir,
                                 radius_mask=2.1,
                                 center_scan=cones_img.center_fitted,
                                 rotations=(60, 120, 180, 240, 300))
    cones_img.get_cone(output_dir=output_dir, radius_mask=2.1)
    cones_img.plot_cones(output_dir=output_dir, radius_mask=2.1)
    cones_img.plot_cones_presence(output_dir=output_dir,
                                  radius_mask=15.5)
    return cones_img.pixels_pos_predict


if __name__ == '__main__':
    pos_predict, pos_true = get_cone_position_simu(output_dir='./tests/resources/')
   # pos_predict = get_cones_position('./tests/resources/cones_1509411741.fits', output_dir='./tests/resources/')
