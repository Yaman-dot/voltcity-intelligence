import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
class Fuzzy:
    def __init__(self, urgency, budget, predict_cost, predict_time):
        self.urg = np.linspace(0,10,100)
        self.bud = np.linspace(0,10,100)
        self.cost = np.arange(0,35.1, 0.1)
        self.time = np.arange(0,220.1, 0.1)
        self.comfort_score = np.linspace(0,100, 1001)

        self.urgency = urgency
        self.budget = budget
        self.predict_Cost = predict_cost
        self.predict_Time = predict_time
        
        self.define_mfs_degrees(self.urg, self.bud, self.cost, self.time, self.comfort_score, self.urgency, self.budget, self.predict_Cost, self.predict_Time)
        
        self.define_rules()
        #self.plot_defuzz()
    def define_mfs_degrees(self, urg, bud, cost, time, comfort_score, urgency, budget, predict_Cost, predict_Time):
        #input urgency
        self.urg_low = fuzz.trapmf(urg, [0, 0, 2, 4])
        self.urg_medium = fuzz.trimf(urg, [3, 5, 7])
        self.urg_high = fuzz.trapmf(urg, [6, 6, 10, 10])

        #input budget
        self.bud_low = fuzz.trapmf(bud, [0, 0, 2, 4])
        self.bud_medium = fuzz.trimf(bud, [3, 5, 7])
        self.bud_high = fuzz.trapmf(bud, [6, 6, 10, 10])
        
        #input cost
        self.cost_low = fuzz.trapmf(cost, [0, 0, 4, 7])
        self.cost_medium = fuzz.trimf(cost, [6, 10, 15])
        self.cost_high = fuzz.trapmf(cost, [14, 18, 30, 30])
        
        #input time
        self.time_low = fuzz.trapmf(time, [0, 0, 30, 60])
        self.time_medium = fuzz.trimf(time, [55, 85, 105])
        self.time_high = fuzz.trapmf(time, [100, 135, 190, 215])
        
        #output comfort score
        self.comfort_low = fuzz.trapmf(comfort_score, [0,0,20,40])
        self.comfort_medium = fuzz.trimf(comfort_score, [30, 50, 70])
        self.comfort_high = fuzz.trapmf(comfort_score, [60, 60, 100, 100])
        
        #membership degrees
        #budget
        self.bl_lvl = fuzz.interp_membership(bud, self.bud_low, budget)
        self.bm_lvl = fuzz.interp_membership(bud, self.bud_medium, budget)
        self.bh_lvl = fuzz.interp_membership(bud, self.bud_high, budget)
        #urgency
        self.ul_lvl = fuzz.interp_membership(urg, self.urg_low, urgency)
        self.um_lvl = fuzz.interp_membership(urg, self.urg_medium, urgency)
        self.uh_lvl = fuzz.interp_membership(urg, self.urg_high, urgency)
        #time
        self.tl_lvl = fuzz.interp_membership(time, self.time_low, predict_Time)
        self.tm_lvl = fuzz.interp_membership(time, self.time_medium, predict_Time)
        self.th_lvl = fuzz.interp_membership(time, self.time_high, predict_Time)
        #cost
        self.cl_lvl = fuzz.interp_membership(cost, self.cost_low, predict_Cost)
        self.cm_lvl = fuzz.interp_membership(cost, self.cost_medium, predict_Cost)
        self.ch_lvl = fuzz.interp_membership(cost, self.cost_high, predict_Cost)
        """ print(f
        Budget Levels:   Low={self.bl_lvl:.3f}, Medium={self.bm_lvl:.3f}, High={self.bh_lvl:.3f}
        Urgency Levels:  Low={self.ul_lvl:.3f}, Medium={self.um_lvl:.3f}, High={self.uh_lvl:.3f}
        Time Levels:     Low={self.tl_lvl:.3f}, Medium={self.tm_lvl:.3f}, High={self.th_lvl:.3f}
        Cost Levels:     Low={self.cl_lvl:.3f}, Medium={self.cm_lvl:.3f}, High={self.ch_lvl:.3f}
        ) """
        return (self.bl_lvl, self.bm_lvl, self.bh_lvl, 
                self.ul_lvl, self.um_lvl, self.uh_lvl,
                self.tl_lvl, self.tm_lvl, self.th_lvl, 
                self.cl_lvl, self.cm_lvl, self.ch_lvl)
    def define_rules(self):
        #low
        #r1: if budget sensitivity is low, and urgency is low, then comfort it high
        self.r1 = np.fmin(np.fmin(self.bl_lvl, self.ul_lvl), self.comfort_high)
        
        #r2: if budget sensitivity is high, and urgency is high then comfort is low
        self.r2 = np.fmin(np.fmin(self.bh_lvl, self.uh_lvl), self.comfort_low)
        
        #r3: if budget is low and cost is high then the comfort is low too
        self.r3 = np.fmin(np.fmin(self.bh_lvl, self.cl_lvl), self.comfort_low)
        
        #r3: if time is high, and urgency is high then comfort is low
        self.r4 = np.fmin(np.fmin(self.th_lvl, self.uh_lvl), self.comfort_low)
        
        
        #medium
        #r4: if budget sensitivity is medium, and urgency is medium, then comfort it medium
        self.r5 = np.fmin(np.fmin(self.bm_lvl, self.um_lvl), self.comfort_medium)
        
        #r6: if budget sensitivity is high, and cost is medium then comfort is medium
        self.r6 = np.fmin(np.fmin(self.bh_lvl, self.ch_lvl), self.comfort_medium)
        
        #r7: if urgency is medium and time is medium then the comfort is medium too
        self.r7 = np.fmin(np.fmin(self.um_lvl, self.tm_lvl), self.comfort_medium)
        
        #r8: if budget is low and cost is low, then comfort is medium
        self.r8 = np.fmin(np.fmin(self.th_lvl, self.uh_lvl), self.comfort_medium)
        
        #r9: urgency is low and time is high then comfort is medium
        self.r9 = np.fmin(np.fmin(self.ul_lvl, self.th_lvl), self.comfort_medium)
        
        #r10: if cost is high and budget is medium then comfort is medium
        self.r10 = np.fmin(np.fmin(self.ch_lvl, self.bm_lvl), self.comfort_medium)
        
        
        #high
        #r11: if budget is high and cost is low then comfort is high
        self.r11 = np.fmin(np.fmin(self.bh_lvl, self.cl_lvl), self.comfort_high)
        
        #r12: if urgency is low and time is low then comfort is high
        self.r12 = np.fmin(np.fmin(self.ul_lvl, self.tl_lvl), self.comfort_high)
        
        #r13: if budget is medium and cost is low then comfort is high
        self.r13 = np.fmin(np.fmin(self.bm_lvl, self.cl_lvl), self.comfort_high)

        #r14: if urgency is low and cost is low then comfort is high
        self.r14 = np.fmin(np.fmin(self.ul_lvl, self.cl_lvl), self.comfort_high)
        
        #r15: if urgency is high and cost is high then comfort is low
        self.r15 = np.fmin(np.fmin(self.uh_lvl, self.ch_lvl), self.comfort_low)
        
        #r16: if budget is low, urgency is medium and time is mediun then comfort is medium
        self.r16 = np.fmin(np.fmin(self.tm_lvl, np.fmin(self.bl_lvl, self.um_lvl)), self.comfort_medium)
        #aggregate everything
        self.aggregated = np.fmax(
            np.fmax(
                np.fmax(
                    np.fmax(self.r1, self.r2),
                    np.fmax(self.r3, self.r4)
                ),
                np.fmax(
                    np.fmax(self.r5, self.r6),
                    np.fmax(self.r7, self.r8)
                )
            ),
            np.fmax(
                np.fmax(
                    np.fmax(self.r9, self.r10),
                    np.fmax(self.r11, self.r12)
                ),
                np.fmax(
                    np.fmax(self.r13, self.r14),
                    np.fmax(self.r15, self.r16)
                )
            )
        )
    """ def plot_defuzz(self):
        #defuzz
        if np.all(self.aggregated == 0):
            centroid = None
            mom = None
        else:
            centroid = fuzz.defuzz(self.comfort_score, self.aggregated, "centroid")
            mom = fuzz.defuzz(self.comfort_score, self.aggregated, "mom")
        print(f"Comfort Score (Centroid): {None if centroid is None else round(centroid, 2)}")
        print(f"Comfort score (mom) {None if mom is None else round(mom,2)}")
        # --- Plot base output MFs and aggregated result
        plt.figure(figsize=(7,4))
        plt.plot(self.comfort_score, self.comfort_low, '--', label='Low (base)')
        plt.plot(self.comfort_score, self.comfort_medium,      '--', label='Medium (base)')
        plt.plot(self.comfort_score, self.comfort_high,      '--', label='High (base)')
        plt.fill_between(self.comfort_score, 0, self.aggregated, alpha=0.6, label='Aggregated Output')
        plt.title('Aggregated Output: Comfort Score')
        plt.xlabel('Comfort (0–10)')
        plt.ylabel('Membership')
        plt.legend()
        plt.grid(True)
        plt.show() """
    def defuzz(self):
        if np.all(self.aggregated == 0):
            self.centroid = None
            self.mom = None
        else:
            self.centroid = fuzz.defuzz(self.comfort_score, self.aggregated, "centroid")
            self.mom = fuzz.defuzz(self.comfort_score, self.aggregated, "mom")
        return self.mom
