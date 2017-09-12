import os

import us.utils.picmus

PICMUS17_PATH = os.path.join('datasets', 'picmus17')
PICMUS16_PATH = os.path.join('datasets', 'picmus16')

# Download PICMUS 2017 data
us.utils.picmus.download_2017(export_path=PICMUS17_PATH,
                              signal_selection=['rf'],
                              transmission_selection=['transmission_1'],
                              pw_number_selection=[1])

# Download PICMUS 2016 experimental data
us.utils.picmus.download_in_vivo_2016(export_path=PICMUS16_PATH)
