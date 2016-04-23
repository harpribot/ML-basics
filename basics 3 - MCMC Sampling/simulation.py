import numpy as np
from random import gauss
from scipy.stats import norm,multivariate_normal
import matplotlib.pyplot as plt


class Sampling:
    def __init__(self,num_samples):
        self.total_samples = num_samples

    def metroplis_1D(self,proposal_var,total_samples=1000):
        self.sampling_method = "metropolis"
        self.total_samples = total_samples
        self.generated_samples = np.zeros(self.total_samples)
        self.proposal_var = proposal_var
        # first sample x
        x = 0
        self.generated_samples[0] = x

        for i in range(1,self.total_samples):
            # Proposal Distribution is N(0,sigma)
            innov = gauss(0,self.proposal_var)
            can = x + innov
            ## the actual distribution is p(x) = 0.3 N(-25,10) + 0.7 N(20,10)
            self.dist_1 = norm(-25,10)
            self.dist_2 = norm(20,10)
            p_can = 0.3 * self.dist_1.pdf(can) + 0.7 * self.dist_2.pdf(can)
            p_x = 0.3 * self.dist_1.pdf(x) + 0.7 * self.dist_2.pdf(x)
            aprob = min(1, p_can/p_x)

            u = np.random.rand()

            if(u < aprob):
                x = can

            self.generated_samples[i] = x
        # Code to check the mean of the distribution
        print 'For Proposal Variance', proposal_var
        print 'Mean of Sampled Distribution', (np.mean(self.generated_samples));

    def metroplis_2D(self,proposal_var,total_samples=1000):
        self.sampling_method = "metropolis"
        self.total_samples = total_samples
        self.generated_samples_x = np.zeros(self.total_samples)
        self.generated_samples_y = np.zeros(self.total_samples)
        self.proposal_var = proposal_var
        # first sample x
        x = 0
        y = 0
        self.generated_samples_x[0] = x
        self.generated_samples_y[0] = y

        for i in range(1,self.total_samples):
            # Proposal Distribution is N(0,kI)
            innov = np.random.multivariate_normal(np.array([0,0]),\
                                            self.proposal_var * np.eye(2),1)

            can_x = x + innov[0,0]
            can_y = y + innov[0,1]
            ## the actual distribution is p(x) = 0.3 N(-25,10) + 0.7 N(20,10)
            self.dist_1 = multivariate_normal(np.array([-25,-25]),10*np.eye(2))
            self.dist_2 = multivariate_normal(np.array([20,20]),10 * np.eye(2))
            p_can = 0.3 * self.dist_1.pdf(np.array([can_x,can_y])) + \
                    0.7 * self.dist_2.pdf(np.array([can_x,can_y]))
            p_x = 0.3 * self.dist_1.pdf(np.array([x,y])) + \
                  0.7 * self.dist_2.pdf(np.array([x,y]))
            aprob = min(1, p_can/p_x)

            u = np.random.rand()

            if(u < aprob):
                x = can_x
                y = can_y

            self.generated_samples_x[i] = x
            self.generated_samples_y[i] = y

    def gibbs_sampling_2D(self,total_samples, T):
        self.sampling_method = "gibbs sampling"
        self.total_samples = total_samples
        self.T = T
        # Initialize the sample storage
        self.generated_samples_x = np.zeros(self.total_samples)
        self.generated_samples_y = np.zeros(self.total_samples)

        self.dist_1 = norm(-25,10)
        self.dist_2 = norm(20,10)

        for i in range(0,self.total_samples):
            # Initialization
            x = 0.0
            y = 0.0
            for j in range(0,self.T):
                # Ratio for x
                a = self.dist_1.pdf(y)
                b = self.dist_2.pdf(y)
                first = 0.3 * (a/(a + b))
                second = 0.7 * (b/(a + b))
                u = np.random.rand()
                if(u < first/(first + second)):
                    x = gauss(-25,10)
                else:
                    x = gauss(20,10)
                # Ratio for y
                a = self.dist_1.pdf(x)
                b = self.dist_2.pdf(x)
                first = 0.3 * (a/(a + b))
                second = 0.7 * (b/(a + b))
                u = np.random.rand()
                if(u < first/(first + second)):
                    y = gauss(-25,10)
                else:
                    y = gauss(20,10)

            self.generated_samples_x[i] = x
            self.generated_samples_y[i] = y


    def plot_histogram_1D(self):
        n, bins, patches = plt.hist(self.generated_samples, \
                                        50,normed=1, facecolor='g', alpha=0.75)
        x = np.sort(self.generated_samples)

        actual_plot = 0.3 * self.dist_1.pdf(x) + \
                            0.7 * self.dist_2.pdf(x)
        plt.plot(x, actual_plot)
        plt.xlabel('Sample')
        plt.ylabel('Probability')
        if self.sampling_method == "gibbs sampling":
            plt.title('Histogram of samples for p(x) = 0.3 N(-25,10) + \
            0.7 N(20,10) for T=%d' %(self.T))
        elif self.sampling_method == "metropolis":
            plt.title('Histogram of samples for p(x) = 0.3 N(-25,10) + \
            0.7 N(20,10) for Var=%d'%(self.proposal_var))
        plt.grid(True)

    def plot_histogram_2D(self):
        plt.hist2d(self.generated_samples_x,self.generated_samples_y,\
                        bins=50,range=np.array([[-50,50],[-50,50]]),normed=1)
        plt.xlabel('Sample')
        plt.ylabel('Probability')
        if self.sampling_method == "gibbs sampling":
            plt.title('Histogram of samples for p(x) = 0.3 N(-25,10) + \
            0.7 N(20,10) for T=%d'%(self.T))
        elif self.sampling_method == "metropolis":
            plt.title('Histogram of samples for p(x) = 0.3 N(-25,10) + \
            0.7 N(20,10) for Var=%d'%(self.proposal_var))
        plt.grid(True)


def main():
    ########### Part 1 & Part 2 - 1D gaussian - Metropolis ###############
    print('Starting Matropolis on 1D gaussian mixture')
    sampler = Sampling(1000)
    plt.figure()
    sampler.metroplis_1D(1)
    sampler.plot_histogram_1D()
    plt.figure()
    sampler.metroplis_1D(10)
    sampler.plot_histogram_1D()
    plt.figure()
    sampler.metroplis_1D(20)
    sampler.plot_histogram_1D()
    plt.figure()
    sampler.metroplis_1D(100)
    sampler.plot_histogram_1D()
    plt.figure()
    sampler.metroplis_1D(400)
    sampler.plot_histogram_1D()
    plt.figure()
    sampler.metroplis_1D(1000)
    sampler.plot_histogram_1D()
    print('1D metropolis completed\n\n')
    ########### Part 3 - 2D gaussian mixture - Metropolis ################
    print('Starting Matropolis on 2D gaussian mixture')
    plt.figure()
    sampler.metroplis_2D(100,30000)
    sampler.plot_histogram_2D()
    plt.figure()
    sampler.metroplis_2D(300,30000)
    sampler.plot_histogram_2D()
    plt.figure()
    sampler.metroplis_2D(500,30000)
    sampler.plot_histogram_2D()
    plt.figure()
    sampler.metroplis_2D(700,30000)
    sampler.plot_histogram_2D()
    print('2D metropolis completed\n\n')
    ############ Part 4 - 2D gaussian mixture - Gibbs Sampling ############
    print('Starting Gibbs Sampling on 2D Gaussian mixture')
    plt.figure()
    sampler.gibbs_sampling_2D(7000,300)
    sampler.plot_histogram_2D()
    print('Gibbs sampling completed\n\n')

    print('Plotting all the plots')
    plt.show()
    print('All done....exiting')


if __name__=="__main__":
    main()
