from digicampipe.image.cones_image import ConesImage, cones_simu


def get_cone_position_simu(output_filename=None):
    test_image, true_positions = cones_simu(
        offset=(-4.3, -2.1),
        angle_deg=10,
        pixel_radius=35,
        output_filename=output_filename,
    )
    cones_img = ConesImage(test_image)
    cones_img.plot_cones(output_filename='tests/resources/cones.png')
    cones_img.plot_fft_cones(output_filename='tests/resources/cones-fft.png')
    cones_img.get_cones_separation_reciprocal(
        output_filename='tests/resources/cones-sep-reciprocal.png'
    )
    cones_img.plot_fft_cones(
        output_filename='tests/resources/cones-fft-masked.png',
        radius_mask=2.1
    )
    cones_img.get_cone(output_filename='tests/resources/cone.png',
                       radius_mask=2.1)
    cones_img.scan_cone_position(
        output_filename='tests/resources/hexagonalicity.png',
        radius_mask=2.1,
        center_scan=cones_img.center_fitted,
        rotations=(60, 120, 180, 240, 300)
    )
    cones_img.plot_cones(output_filename='tests/resources/cones-filtered.png',
                         radius_mask=2.1)
    cones_img.plot_cones_presence(
        output_filename='tests/resources/cones-presence.png'
    )
    cones_img.fit_camera_geometry()
    cones_img.refine_camera_geometry()
    cones_img.plot_camera_geometry(
        output_filename='tests/resources/cones-presence-filtered.png'
    )
    return cones_img.pixels_pos_predict, true_positions


def get_cones_position(filename, threshold_std=1):
    from astropy.io import fits

    image = fits.open(filename)[0].data
    cones_img = ConesImage(image, threshold_std=threshold_std)
    cones_img.plot_cones(
        output_filename=filename.replace('.fits','-cones.png')
    )
    cones_img.plot_fft_cones(
        output_filename=filename.replace('.fits','-cones-fft.png')
    )
    cones_img.get_cones_separation_reciprocal(
        output_filename=filename.replace('.fits','-cones-sep-reciprocal.png')
    )
    cones_img.plot_fft_cones(
        output_filename=filename.replace('.fits','-cones-fft-masked.png'),
        radius_mask=3.1
    )
    cones_img.get_cone(output_filename=filename.replace('.fits','-cone.png'),
                       radius_mask=3.1,
                       cone_filename=filename.replace('.fits','-cone.fits'),
                       average_rotation=True)
    """
    cones_img.scan_cone_position(
        output_filename=filename.replace('.fits','-hexagonalicity.png'),
        radius_mask=5.1,
        center_scan=cones_img.center_fitted,
        rotations=(60, 120, 180, 240, 300)
    )
    """
    cones_img.plot_cones(
        output_filename=filename.replace('.fits','-cones-filtered.png'),
        radius_mask=3.1
    )
    cones_img.plot_cones_presence(
        output_filename=filename.replace('.fits','-cones-presence.png')
    )
    cones_img.fit_camera_geometry(radius_mask=10.1,
                                  sigma_peak=2.5,
                                  offset_max=7)
    cones_img.refine_camera_geometry()
    cones_img.plot_camera_geometry(
        output_filename=filename.replace('.fits','-cones-presence-filtered.png')
    )
    #cones_img.save_camera_geometry('../data/position_pred.txt', '../data/position_fit.txt')
    cones_img.save_camera_geometry('tests/resources/position_pred.txt', 'tests/resources/position_fit.txt')

    return cones_img.pixels_pos_predict


if __name__ == '__main__':
    """
    pos_predict, pos_true = get_cone_position_simu(
        output_filename='./tests/resources/cones-orig.png'
    )
    """
    from pkg_resources import resource_filename
    example_lid_CCD_image_file_path = resource_filename('digicampipe',
                                                        'tests/resources/cones_1509411741.fits')
    pos_predict = get_cones_position(example_lid_CCD_image_file_path, threshold_std=1)
    """
    get_cones_position("../data/camera_2017-01-16.fits", threshold_std=15)
    """