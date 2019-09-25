

class ASCReader:

    def __init__ (self, fileName):
        self.FileName = fileName
        self.Options  = self.parse_config ( )
        self.Time, self.Amplitude = self.parse_data();

    def getParameterByName (self, Name ):
        return self.Options[Name]

    def getTime ( self, us=False ):
        if  us:
            return self.Time*1e6
        else:
            return self.Time

    def getData ( self ):
        return self.Amplitude

    def getGain ( self, Linear=False ):
        if Linear:
            return 10 ** ( float( self.Options['Gain'] ) / -20)
        else:
            return float( self.Options['Gain'] )

    def parse_data ( self ):
        data = np.loadtxt( self.FileName, skiprows=72, encoding='ISO-8859-1' )
        dt = float ( self.Options ['TimeBase'] ) * 1e-6
        dtO = float ( self.Options ['TriggerDelay'] ) *1e-6
        amp = 100 * data / 2048
        t = np.arange ( 0, (data.shape[0]*dt), dt) + dtO
        return t, amp

    def parse_config(self):
        COMMENT_CHAR = '#'
        OPTION_CHAR =  ':'

        options = {}
        f = open(self.FileName, encoding='latin9')
        for line in f:
            # First, remove comments:
            if COMMENT_CHAR in line:
                # split on comment char, keep only the part before
                line, comment = line.split(COMMENT_CHAR, 1)
            # Second, find lines with an option=value:
            if OPTION_CHAR in line:
                # split on option char:
                option, value = line.split(OPTION_CHAR, 1)
                # strip spaces:
                strOption = "".join(option)
                if '[' in strOption:
                    option, ba = strOption.split ( '[', 1 )

                option = option.strip()
                # print ( option )
                if option == 'Bemerkungen':
                    param_list = value.split(',')
                    for opt in param_list:
                        #print ( opt.strip() )
                        if '=' in opt:
                            param, val = "".join(opt).split('=', 1)
                            options[param.strip() ] = val.strip()
                            #print ('%s = %s' % (param, val ) )

                value = value.strip()
                # store in dictionary:
                options[option] = value
        f.close()
        return options
