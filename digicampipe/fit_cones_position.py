from digicampipe.image.cones_image import ConesImage


def get_cone_position_simu():
    cones_img = ConesImage('test', output_dir='./tests/resources/')
    cones_img.plot_cones(output_dir='./tests/resources/')
    cones_img.plot_fft_cones(output_dir='./tests/resources/')
    cones_img.get_cones_separation_reciprocal(output_dir='./tests/resources/')
    cones_img.plot_fft_cones(output_dir='./tests/resources/', radius_mask=2.1)
    cones_img.scan_cone_position(output_dir='./tests/resources/',
                                 radius_mask=2.1,
                                 center_scan=cones_img.center_fitted,
                                 rotations=(60, 120, 180, 240, 300))
    cones_img.get_cone(output_dir='./tests/resources/', radius_mask=2.1)
    cones_img.plot_cones(output_dir='./tests/resources/', radius_mask=2.1)
    cones_img.plot_cones_presence(output_dir='./tests/resources/',
                                  radius_mask=15.5)


def get_cones_position():
    cones_img = ConesImage('./tests/resources/cones_1509411741.fits')
    cones_img.plot_cones(output_dir='./tests/resources/')
    cones_img.plot_fft_cones(output_dir='./tests/resources/')
    cones_img.get_cones_separation_reciprocal(output_dir='./tests/resources/')
    cones_img.plot_fft_cones(output_dir='./tests/resources/', radius_mask=2.1)
    cones_img.scan_cone_position(output_dir='./tests/resources/',
                                 radius_mask=2.1,
                                 center_scan=cones_img.center_fitted,
                                 rotations=(60, 120, 180, 240, 300))
    cones_img.get_cone(output_dir='./tests/resources/', radius_mask=2.1)
    cones_img.plot_cones(output_dir='./tests/resources/', radius_mask=2.1)
    cones_img.plot_cones_presence(output_dir='./tests/resources/',
                                  radius_mask=15.5)


if __name__ == '__main__':
    get_cone_position_simu()
    get_cones_position()
