from . import Processor


class HillasToText(Processor):
    def __init__(self, output_filename):
        self.output_filename
        self.ofile = open(output_filename, 'w')
        self.ofile.write(
            "# size cen_x cen_y length width r phi psi miss "
            "skewness kurtosis event_number timestamp border time spread\n")

    def __call__(self, event):

        size         = event.dl2.shower.size
        cen_x        = event.dl2.shower.cen_x.value
        cen_y        = event.dl2.shower.cen_y.value
        length       = event.dl2.shower.length.value
        width        = event.dl2.shower.width.value
        r            = event.dl2.shower.r.value
        phi          = event.dl2.shower.phi.value
        psi          = event.dl2.shower.psi.value
        miss         = event.dl2.shower.miss.value
        skewness     = event.dl2.shower.skewness
        kurtosis     = event.dl2.shower.kurtosis
        event_number = event.r0.event_id
        border       = 1 if event.dl1.tel[1].on_border else 0
        time_spread  = event.dl1.tel[1].time_spread
        time_stamp = event.r0.tel[1].local_camera_clock

        self.ofile.write(
            str(size) + " " +
            str(cen_x) + " " +
            str(cen_y) + " " +
            str(length) + " " +
            str(width) + " " +
            str(r) + " " +
            str(phi) + " " +
            str(psi) + " " +
            str(miss) + " " +
            str(skewness) + " " +
            str(kurtosis) + " " +
            str(event_number) + " " +
            str(time_stamp) + " " +
            str(border) + " " +
            str(time_spread) + "\n"
        )
        self.ofile.flush()

    def __del__(self):
        self.ofile.close()
