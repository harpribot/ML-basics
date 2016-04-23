import numpy as np
import matplotlib.pyplot as mpl
from scipy import linalg, io
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit
from scipy.io.wavfile import write

"""
    @author Harshal Priyadarshi - UTEid - hp7325
    @Topic - Independent Component Analysis
"""
##### Independent Component Analysis class ######
class ICA:
    def __init__(self,input_file, num_iter,sig_choice = []):
        self.input_file = input_file
        self.sig_choice = sig_choice
        self.load_data()
        self.mix_signal()
        self.random_Weight()
        self.num_iter = num_iter
        self.alpha = 0.01


    def load_data(self):
        data = io.loadmat(self.input_file)
        self.data = data
        if data.has_key('U'):
            self.U = data['U']
            self.num_sig = self.U.shape[0]
            self.A = data['A']
        elif data.has_key('sounds'):
            self.U = data['sounds']
            self.getRelevantSignals()
            self.num_sig = self.U.shape[0]
            self.A = np.random.rand(self.num_sig, self.num_sig)
        else:
            raise Exception('Invalid file input. It lacks the desired key tokens')


    def getRelevantSignals(self):
        self.U = self.U[self.sig_choice];

    def mix_signal(self):
        self.X = np.dot(self.A, self.U)

    def algorithm_W(self):
        for i in range(0,self.num_iter):
            self.ICA_iteration_W()

    def ICA_iteration_W(self):
        # Get the current estimate
        Y = np.dot(self.W,self.X)
        # Get the Cumulative Distribution Function, Z - Sigmoid function
        Z = expit(Y)
        # Get the update
        # We have multiplied original del_W with (W.T * W), to ensure that alpha reduces as
        # W minimizes. This ensures proper convergence.
        del_W = self.alpha * np.dot((np.identity(self.num_sig) + np.dot((1 - 2*Z), Y.T)),self.W)
        self.W += del_W

    def algorithm_W_beta(self):
        self.beta = np.random.uniform(0.8, 1.2, (self.num_sig,1))
        self.beta_iter = 8
        for i in range(self.beta_iter):
            print(i)
            for j in range(self.num_iter):
                self.ICA_iteration_W_beta()
            self.update_beta()


    def ICA_iteration_W_beta(self):
        # Get the current estimate
        Y = np.dot(self.W_beta,self.X)
        # Get the Cumulative Distribution Function, Z - Sigmoid function
        Z = expit(np.multiply(self.beta,Y))
        # Get the update
        del_W = self.alpha * np.dot((np.identity(self.num_sig) + np.dot(np.multiply(self.beta,(1 - 2*Z)), Y.T)),self.W_beta)
        self.W_beta += del_W

    def update_beta(self):
        Y = np.dot(self.W_beta,self.X)
        Z = expit(np.multiply(self.beta,Y))
        self.beta += self.alpha * np.ndarray.sum(np.multiply(Y,(1 - 2 * Z)),axis = 1, keepdims=True)/5


    def recover_signal_W(self):
        U_recovered = np.dot(self.W,self.X)
        # Scale the recovered signal
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.U_recovered = scaler.fit_transform(U_recovered.T).T


    def recover_signal_W_beta(self):
        U_recovered = np.dot(self.W_beta,self.X)
        # Scale the recovered signal
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.U_recovered_beta = scaler.fit_transform(U_recovered.T).T

    def plot_signals_W(self):
        mpl.figure()
        if self.data.has_key('sounds'):
            mpl.ylim([-1,20])
        plot_points = self.U.shape[1]
        x = range(plot_points)
        for i in range(0, self.num_sig):
            if i ==1:
                label_original = 'original'
                label_mixed = 'mixed'
                label_reconst = 'reconstructed'
            else:
                label_original = None
                label_mixed = None
                label_reconst = None
            mpl.plot(x, self.U[i] + i + 0.5, 'b',label=label_original)
            mpl.plot(x, self.X[i] + self.num_sig + i + 0.5, 'g', label=label_mixed)
            mpl.plot(x, self.U_recovered[i] +  2 * self.num_sig + i + 0.5, 'r', label=label_reconst)

        mpl.xlabel('Signals')
        mpl.ylabel('Amplitude(y-shifted)')
        mpl.title('Reconstrcution Plot for Bigger Signal')
        mpl.legend()


    def plot_signals_W_beta(self):
        mpl.figure()
        if self.data.has_key('sounds'):
            mpl.ylim([-1,20])
        plot_points = self.U.shape[1]
        x = range(plot_points)
        for i in range(0, self.num_sig):
            if i ==1:
                label_original = 'original'
                label_mixed = 'mixed'
                label_reconst = 'reconstructed'
            else:
                label_original = None
                label_mixed = None
                label_reconst = None
            mpl.plot(x, self.U[i] + i + 0.5, 'b',label=label_original)
            mpl.plot(x, self.X[i] + self.num_sig + i + 0.5, 'g', label=label_mixed)
            mpl.plot(x, self.U_recovered_beta[i] +  2 * self.num_sig + i + 0.5, 'r', label=label_reconst)

        mpl.xlabel('Signals')
        mpl.ylabel('Amplitude(y-shifted)')
        mpl.title('Reconstrcution Plot for Bigger Signal with beta update')
        mpl.legend()


    def random_Weight(self):
        self.W = np.random.uniform(0.0, 0.1, (self.num_sig,self.num_sig))
        self.W_beta = self.W

    def store_original_sound(self):
        for i in range(self.num_sig):
            data = self.U[i]
            scaled = np.int16(data/np.max(np.abs(data)) * 32767)
            file_name = 'signals-big_data/original/original_%d.wav' %(i)
            write(file_name, 11025, scaled)

    def store_recovered_sound(self):
        for i in range(self.num_sig):
            data = self.U_recovered[i]
            scaled = np.int16(data/np.max(np.abs(data)) * 32767)
            file_name = 'signals-big_data/recovered_no_beta/recovered_%d.wav' %(i)
            write(file_name, 11025, scaled)

    def store_recovered_sound_W_beta(self):
        for i in range(self.num_sig):
            data = self.U_recovered_beta[i]
            scaled = np.int16(data/np.max(np.abs(data)) * 32767)
            file_name = 'signals-big_data/recovered_beta/recovered_%d.wav' %(i)
            write(file_name, 11025, scaled)

    def find_correlation(self):
        self.correlation_matrix = np.zeros([self.num_sig, self.num_sig])
        for i in range(0, self.num_sig):
            for j in range(0, self.num_sig):
                self.correlation_matrix[i,j] = np.corrcoef(self.U[i],self.U_recovered[j])[0,1]

    def find_correlation_beta(self):
        self.correlation_matrix = np.zeros([self.num_sig, self.num_sig])
        for i in range(0, self.num_sig):
            for j in range(0, self.num_sig):
                self.correlation_matrix[i,j] = np.corrcoef(self.U[i],self.U_recovered_beta[j])[0,1]

    def print_correlations(self):
        print self.correlation_matrix

##### main function #####
def main():
    ##### Get the data
    input_small = 'data/small_data.mat'
    num_iter_sm = 100000
    # Change this to run ICA on different signals for bigger data (BEGINS AT 0)
    relevant_sig = [1,2,3]
    #### For Smaller Signal
    # ICA algorithm without beta update on small signal
    ica_small = ICA(input_small,num_iter_sm)
    ica_small.algorithm_W()
    ica_small.recover_signal_W()
    ica_small.plot_signals_W()
    ica_small.find_correlation()
    print "Printing Correlation for the smaller data"
    ica_small.print_correlations()

    # ICA algorithm with beta update on small signal (UNCOMMENT TO RUN)
    '''
    ica_small.algorithm_W_beta()
    ica_small.recover_signal_W_beta()
    ica_small.plot_signals_W_beta()
    '''
    #### For Bigger Signal
    # ICA algorithm without beta update on bigger signal
    input_big = 'data/big_data.mat'
    num_iter_bg = 10000
    ica_big = ICA(input_big, num_iter_bg,relevant_sig)
    ica_big.algorithm_W()
    ica_big.recover_signal_W()
    ica_big.find_correlation()
    print "Printing Correlation for the larger data without beta update"
    ica_big.print_correlations()
    ica_big.plot_signals_W()
    ica_big.store_original_sound()
    ica_big.store_recovered_sound()

    # ICA algorithm with beta update on bigger signal (UNCOMMENT TO RUN)
    '''
    ica_big.algorithm_W_beta()
    ica_big.recover_signal_W_beta()
    ica_big.plot_signals_W_beta()
    ica_big.store_recovered_sound_W_beta()
    ica_big.find_correlation_beta()
    print "Printing Correlation for the larger data with beta update"
    ica_big.print_correlations()
    '''
    #### Show all the plots
    mpl.show()

if __name__ == "__main__":
    main()
