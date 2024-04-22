import numpy as np
from cores.configuration.configuration import Configuration
config = Configuration()

class BoundingShapeCoef(object):

    def __init__(self):
        super(BoundingShapeCoef, self).__init__()

        self.coefs = {}
        self.coefs["LINK3_BB"] = np.array([[1/(0.09*0.09), 0.0, 0.0],
                                             [0.0, 1/(0.09*0.09), 0.0],
                                             [0.0, 0.0, 1/(0.165*0.165)]], dtype=config.np_dtype)
        
        self.coefs["LINK4_BB"] = np.array([[1/(0.09*0.09), 0.0, 0.0],
                                             [0.0, 1/(0.09*0.09), 0.0],
                                             [0.0, 0.0, 1/(0.15*0.15)]], dtype=config.np_dtype)
        
        self.coefs["LINK5_1_BB"] = np.array([[1/(0.09*0.09), 0.0, 0.0],
                                             [0.0, 1/(0.09*0.09), 0.0],
                                             [0.0, 0.0, 1/(0.14*0.14)]], dtype=config.np_dtype)
        
        self.coefs["LINK5_2_BB"] = np.array([[1/(0.055*0.055), 0.0, 0.0],
                                             [0.0, 1/(0.055*0.055), 0.0],
                                             [0.0, 0.0, 1/(0.125*0.125)]], dtype=config.np_dtype)
        
        self.coefs["LINK6_BB"] = np.array([[1/(0.08*0.08), 0.0, 0.0],
                                             [0.0, 1/(0.08*0.08), 0.0],
                                             [0.0, 0.0, 1/(0.11*0.11)]], dtype=config.np_dtype)
        
        self.coefs["LINK7_BB"] = np.array([[1/(0.07*0.07), 0.0, 0.0],
                                             [0.0, 1/(0.07*0.07), 0.0],
                                             [0.0, 0.0, 1/(0.14*0.14)]], dtype=config.np_dtype)
        
        self.coefs["HAND_BB"] = np.array([[1/(0.07*0.07), 0.0, 0.0],
                                             [0.0, 1/(0.12*0.12), 0.0],
                                             [0.0, 0.0, 1/(0.10*0.10)]], dtype=config.np_dtype)
        
        # self.coefs["HAND_BB"] = np.array([[1/(0.12*0.12), 0.0, 0.0],
        #                                      [0.0, 1/(0.12*0.12), 0.0],
        #                                      [0.0, 0.0, 1/(0.12*0.12)]], dtype=config.np_dtype)

        self.coefs_sqrt = {}
        self.coefs_sqrt["LINK3_BB"] = np.array([[1/0.09, 0.0, 0.0],
                                             [0.0, 1/0.09, 0.0],
                                             [0.0, 0.0, 1/0.165]], dtype=config.np_dtype)
        
        self.coefs_sqrt["LINK4_BB"] = np.array([[1/0.09, 0.0, 0.0],
                                             [0.0, 1/0.09, 0.0],
                                             [0.0, 0.0, 1/0.15]], dtype=config.np_dtype)
        
        self.coefs_sqrt["LINK5_1_BB"] = np.array([[1/0.09, 0.0, 0.0],
                                                [0.0, 1/0.09, 0.0],
                                                [0.0, 0.0, 1/0.14]], dtype=config.np_dtype)
        
        self.coefs_sqrt["LINK5_2_BB"] = np.array([[1/0.055, 0.0, 0.0],
                                                [0.0, 1/0.055, 0.0],
                                                [0.0, 0.0, 1/0.125]], dtype=config.np_dtype)
        
        self.coefs_sqrt["LINK6_BB"] = np.array([[1/0.08, 0.0, 0.0],
                                                [0.0, 1/0.08, 0.0],
                                                [0.0, 0.0, 1/0.11]], dtype=config.np_dtype)
        
        self.coefs_sqrt["LINK7_BB"] = np.array([[1/0.07, 0.0, 0.0],
                                                [0.0, 1/0.07, 0.0],
                                                [0.0, 0.0, 1/0.14]], dtype=config.np_dtype)
        
        self.coefs_sqrt["HAND_BB"] = np.array([[1/0.07, 0.0, 0.0],
                                             [0.0, 1/0.12, 0.0],
                                             [0.0, 0.0, 1/0.10]], dtype=config.np_dtype)
        
        # self.coefs_sqrt["HAND_BB"] = np.array([[1/0.12, 0.0, 0.0],
        #                                      [0.0, 1/0.12, 0.0],
        #                                      [0.0, 0.0, 1/0.12]], dtype=config.np_dtype)
        