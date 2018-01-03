from digicampipe.image.cones_image import ConesImage, cones_simu


def get_cone_position_simu(output_dir=None):
    test_image, true_positions = cones_simu(
        offset=(-4.3, -2.1),
        angle_deg=10,
        pixel_radius=35,
        output_dir=output_dir,
    )
    cones_img = ConesImage(
        test_image,
        output_dir=output_dir,
    )
    cones_img.plot_cones(output_dir=output_dir)
    cones_img.plot_fft_cones(output_dir=output_dir)
    cones_img.get_cones_separation_reciprocal(output_dir=output_dir)
    cones_img.plot_fft_cones(output_dir=output_dir, radius_mask=2.1)
    cones_img.get_cone(output_dir=output_dir, radius_mask=2.1)
    cones_img.scan_cone_position(output_dir=output_dir,
                                 radius_mask=2.1,
                                 center_scan=cones_img.center_fitted,
                                 rotations=(60, 120, 180, 240, 300))
    cones_img.plot_cones(output_dir=output_dir, radius_mask=2.1)
    cones_img.plot_cones_presence(output_dir=output_dir)
    cones_img.fit_camera_geometry()
    cones_img.refine_camera_geometry()
    cones_img.plot_camera_geometry(output_dir=output_dir)
    return cones_img.pixels_pos_predict, true_positions


def get_cones_position(filename, output_dir=None):
    cones_img = ConesImage(filename, './tests/resources/cones_1509411741-cone.fits')
    cones_img.plot_cones(output_dir=output_dir)
    cones_img.plot_fft_cones(output_dir=output_dir)
    cones_img.get_cones_separation_reciprocal(output_dir=output_dir)
    cones_img.plot_fft_cones(output_dir=output_dir, radius_mask=2.1)
    cones_img.get_cone(output_dir=output_dir, radius_mask=2.1)
    cones_img.scan_cone_position(output_dir=output_dir,
                                 radius_mask=2.1,
                                 center_scan=cones_img.center_fitted,
                                 rotations=(60, 120, 180, 240, 300))
    cones_img.plot_cones(output_dir=output_dir, radius_mask=2.1)
    cones_img.plot_cones_presence(output_dir=output_dir)
    cones_img.fit_camera_geometry()
    cones_img.refine_camera_geometry()
    cones_img.plot_camera_geometry(output_dir=output_dir)
    return cones_img.pixels_pos_predict


if __name__ == '__main__':
    pos_predict, pos_true = get_cone_position_simu(output_dir='./tests/resources/')
    pos_predict = get_cones_position('./tests/resources/cones_1509411741.fits', output_dir='./tests/resources/')
