import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# this file contains a main and is hence executable


class Portfolio(object):
    # Simulate price movements for a portfolio of stocks
    def __init__(self, sec1mean, sec2mean, sec1vol, sec2vol, corr, rebalance_threshold):
        self.number_of_stocks = 2
        self.initial_prices = np.asarray([5, 5])
        self.prices = self.initial_prices
        self.initial_holdings = 10 * np.ones((self.number_of_stocks,))
        self.holdings = self.initial_holdings
        self.initial_total = 100
        self.total = self.initial_total
        self.initial_weightings = [.5, .5]
        self.weightings = self.initial_weightings
        # input
        self.means = np.asarray([sec1mean, sec2mean])
        self.corr = corr
        self.sec1_volume = sec1vol
        self.sec2vol = sec2vol
        self.daily_means = self.means / 252
        self.daily_sec1_volume = self.sec1_volume / np.sqrt(252)
        self.dailysec2vol = self.sec2vol / np.sqrt(252)
        dailycov = self.daily_sec1_volume * self.dailysec2vol * self.corr
        self.daily_covariance_matrix = np.asarray(
            [[self.daily_sec1_volume ** 2, dailycov], [dailycov, self.dailysec2vol ** 2]])
        self.rebalance_threshold = rebalance_threshold

    # simulate price movements
    @staticmethod
    def brownian(periods):
        dt = 1
        # standard brownian increment = multivariate_normal distribution * sqrt of dt
        b = np.random.multivariate_normal((0., 0.), ((1., 0.), (0., 1.)), int(periods)) * np.sqrt(dt)
        # standard brownian motion for two variables ~ N(0,t)
        w = np.cumsum(b, axis=0)
        w = np.insert(w, 0, (0., 0.), axis=0)
        w = np.asarray(w)
        return w

    def geometric_brownian_motion(self, w_brownian, _time_period):
        # Simulate GBM as sum of drift and diffusion term for all assets and all prices in the time period
        # this means we assume lognormal price movements
        # w_brownian:      brownian motion
        # time_period:      time period
        price_array = []
        # divide time axis from 0 to 1 into T pieces,
        time_period = np.linspace(0, _time_period, _time_period + 1)
        l_cholesky = np.linalg.cholesky(self.daily_covariance_matrix)
        variance = self.daily_covariance_matrix.diagonal()
        for i in range(_time_period + 1):
            drift = (self.daily_means - (0.5 * variance)) * time_period[i]
            diffusion = np.dot(l_cholesky, w_brownian[i])
            price_array.append(self.initial_prices * np.exp(drift + diffusion))
        price_array = np.asarray(price_array)
        return price_array

    def price_move(self, periods):
        # prices are assumed to move according to GBM
        w = self.brownian(periods)
        return self.geometric_brownian_motion(w, periods)

    def simulate(self, paths, transaction_cost, periods, seed):
        # simulate portfolio performance
        # paths: holds all simulated price paths
        cost = 0
        trade = 0
        number_rebalances = 0
        decrease_return = 0
        fig, ax = plt.subplots(nrows=1, ncols=1)
        np.random.seed(seed)
        for i in range(paths):
            price_movements = self.price_move(periods)
            print("path %d: " % (i + 1))
            trade_path, cost_path, n_rebalance_path, decrease_return_path = self.rebalance(price_movements,
                                                                                           transaction_cost, periods)
            cost += cost_path
            trade += trade_path
            number_rebalances += n_rebalance_path
            decrease_return += decrease_return_path
            t = np.linspace(0, periods, periods + 1)
            image, = ax.plot(t, price_movements[:, 0], label="stock1")
            image, = ax.plot(t, price_movements[:, 1], label="stock2", ls='--')
            plt.ylabel('stock price, $')
            plt.xlabel('time, day')
            plt.title('correlated brownian simulation')
            plt.draw()
            try:
                fig.savefig("simulate.png")
            except BaseException as e:
                print(str(e))
        average_rebalance = number_rebalances / paths
        average_dollar_traded = trade / paths
        average_tcost = cost / paths
        average_decrease_return = decrease_return / paths
        print(
            "average number of rebalances: %.3f\naverage dollars traded: %.3f$\naverage transaction cost as percentage of book value: %.3f%%\nexpected transaction costs: %.3f%%"
            % (average_rebalance, average_dollar_traded, average_tcost * 100, average_decrease_return * 100))

    def rebalance(self, price_movements, transaction_cost, periods):
        # rebalance portfolio so that each asset is weighted with its initial weighting
        # price_movements: pre calculated GBM price movements
        # transaction_cost: Cost of a single transaction on one stock
        # periods:
        trades = []
        price_spread = []
        costs = []
        n_rebalance = 0
        # len(pricemovements) = periods + 1
        for i in range(1, periods + 1):
            new_prices = price_movements[i]
            # update prices, dollar value, and weightings of a portfolio each time prices change
            self.update_prices(new_prices)
            difference = np.subtract(self.weightings, self.initial_weightings)
            # max returns a (positive) percentage difference between the actual weigntings and the desired weightings
            if max(difference) >= self.rebalance_threshold:
                # change the holdings so that the actual weightings are as desired
                self.update_holdings()
                # difference in weightings * total = change of the amount of dollar invested in two stocks
                trade = np.sum(np.absolute(difference * self.total))
                trades.append(trade)
                costs.append(trade * transaction_cost)
                price_spread.append(np.round(self.prices, 2))
                n_rebalance += 1
        data = {"price spread, $": price_spread,
                "size of the trade, $": trades,
                "transaction cost, $": costs}
        df = pd.DataFrame(data=data, index=range(1, n_rebalance + 1))
        df.index.name = "#rebalancing"
        print(df)  # TODO: make log function
        # return metrics
        trade_total = sum(trades)
        cost_total = trade_total * transaction_cost
        annualized_periods = periods / 252
        annualized_return = (self.total / self.initial_total) ** (1 / annualized_periods) - 1
        value_minus_cost = ((self.total - cost_total) / self.initial_total) ** (1 / annualized_periods) - 1
        decrease_return = annualized_return - value_minus_cost
        cost_total_per = cost_total / self.total
        # set parameters back to initial value
        self.reset()
        return trade_total, cost_total_per, n_rebalance, decrease_return

    def reset(self):
        self.weightings = self.initial_weightings
        self.holdings = self.initial_holdings
        self.prices = self.initial_prices
        self.total = self.initial_total

    def update_prices(self, new_prices):
        self.prices = new_prices
        # dot product of the number of shares and price per share gives total portfolio value
        self.total = np.dot(self.holdings, new_prices)
        # the weight of stocks after stock prices change = (number of share * price of stock per share)/total amount of asset
        self.weightings = [holding * price / self.total for price, holding in zip(self.prices, self.holdings)]

    def update_holdings(self):
        # the holdings array gives the number of shares per symbol
        self.holdings = [self.total * initWeight / price for initWeight, price in
                         zip(self.initial_weightings, self.prices)]
        # the weightings array gives the actual contribution of each symbol to the portfolio
        self.weightings = [price * holding / self.total for holding, price in zip(self.holdings, self.prices)]

    # compute how transaction cost vary with respect to other variables
    def decrease_return(self, price_movements, transaction_cost, periods):
        cost_total = 0
        for i in range(1, len(price_movements)):
            new_prices = price_movements[i]
            self.update_prices(new_prices)
            difference = np.subtract(self.weightings, self.initial_weightings)
            if max(difference) >= self.rebalance_threshold:
                self.update_holdings()
                trade = np.sum(np.absolute(difference * self.total))
                cost_total += trade * transaction_cost
        annualizedPeriods = periods / 252
        annualizedReturn = (self.total / self.initial_total) ** (1 / annualizedPeriods) - 1
        postcost = ((self.total - cost_total) / self.initial_total) ** (1 / annualizedPeriods) - 1
        decreaseReturn = annualizedReturn - postcost
        self.reset()
        return decreaseReturn

    def tests(self, paths, transaction_cost, periods, step, seed):
        mean_decrease = []
        total_decrease = 0
        fig, ax = plt.subplots(nrows=1, ncols=1)
        np.random.seed(seed)
        for i in range(1, paths + 1):
            price_movements = self.price_move(periods)
            # return minus transaction cost
            decrease_return = self.decrease_return(price_movements, transaction_cost, periods)
            # why do we multiply with 100?
            total_decrease += decrease_return * 100
            if i % step == 0:
                mean_decrease.append(total_decrease / i)
        print("when seed = %d, paths = %d, the average transaction cost is: %f%%" % (seed, paths, mean_decrease[-1]))
        t = np.linspace(1, paths, len(mean_decrease))
        image, = ax.plot(t, mean_decrease)
        plt.ylabel('sample mean transaction cost (%)')
        plt.xlabel('number of paths')
        plt.title('convergence test (seed = %d)' % (seed))
        plt.draw()
        fig.savefig("convergence test (seed=%d).png" % (seed))

    def update_corr(self, corr):
        dailycov = self.daily_sec1_volume * self.dailysec2vol * corr
        self.daily_covariance_matrix = np.asarray(
            [[self.daily_sec1_volume ** 2, dailycov], [dailycov, self.dailysec2vol ** 2]])

    def update_sec1_volume(self, sec1_volume):
        self.sec1_volume = sec1_volume
        # as we take sqrt(num_days in year) this should be some kind of volatility
        self.daily_sec1_volume = sec1_volume / np.sqrt(252)
        daily_covariance = self.daily_sec1_volume * self.dailysec2vol * self.corr
        self.daily_covariance_matrix = np.asarray(
            [[self.daily_sec1_volume ** 2, daily_covariance], [daily_covariance, self.dailysec2vol ** 2]])

    def update_threshold(self, threshold):
        self.rebalance_threshold = threshold

    def updateSec1Mean(self, sec1mean):
        self.means[0] = sec1mean
        self.daily_means = self.means / 252

    def solveCorr(self, paths, tcost, periods, seed):
        start = 0
        end = 1
        x = np.linspace(0, 1, 11)
        y = []
        for i in range(len(x)):
            totalDecrease = 0
            self.update_corr(x[i])
            np.random.seed(seed)
            for j in range(paths):
                pricemovements = self.price_move(periods)
                decreaseReturn = self.decrease_return(pricemovements, tcost, periods)
                totalDecrease += decreaseReturn * 100
            meanDecrease = np.round(totalDecrease / paths, 1)
            y.append(meanDecrease)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        image, = ax.plot(x, y)
        plt.ylabel('transaction cost (%)')
        plt.xlabel('correlation coefficient')
        plt.title('corr - tcost graph')
        plt.draw()
        fig.savefig('corr-tcost graph')
        print(
            'corr-tcost:\nseed=%d\nsec1vol=%.2f\nsec2vol=%.2f\ncorr=%.2f-%.2f\nsec1mean=%.2f\nsec2mean=%.2f\nthreshold=%.2f'
            % (
                seed, self.sec1_volume, self.sec2vol, start, end, self.means[0], self.means[1],
                self.rebalance_threshold))
        print('coeff:', np.polyfit(x, y, 1))
        '''reg = linear_model.Lasso(alpha = 0.1)
        reg.fit(x,y)
        print('lasso coeff:',reg.coef_)
        print('lasso intercept',reg.intercept_)'''

    def solveSec1Vol(self, paths, tcost, periods, seed):
        start = .01
        end = .51
        x = np.linspace(start, end, 11)
        y = []
        for i in range(len(x)):
            totalDecrease = 0
            self.update_sec1_volume(x[i])
            np.random.seed(seed)
            for j in range(paths):
                pricemovements = self.price_move(periods)
                decreaseReturn = self.decrease_return(pricemovements, tcost, periods)
                totalDecrease += decreaseReturn * 100
            meanDecrease = np.round(totalDecrease / paths, 1)
            y.append(meanDecrease)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        image, = ax.plot(x, y)
        plt.ylabel('transaction cost (%)')
        plt.xlabel('security 1 volatility')
        plt.title('sec1vol - tcost graph')
        plt.draw()
        fig.savefig('sec1vol-tcost graph')
        print(
            'sec1vol_tcost:\nseed=%d\nsec1vol=%.2f-%.2f\nsec2vol=%.2f\ncorr=%.2f\nsec1mean=%.2f\nsec2mean=%.2f\nthreshold=%.2f'
            % (seed, start, end, self.sec2vol, self.corr, self.means[0], self.means[1], self.rebalance_threshold))
        print("coeff:", np.polyfit(x, y, 1))

    def solveSec1Mean(self, paths, tcost, periods, seed):
        # What does it do?
        # paths:
        # tcost: Transaction cost?
        # perdiods:
        # seed: Random seed used for each path
        start = 0
        end = .5
        # span a x grid
        x = np.linspace(0, .5, 11)
        # number of paths to simulate
        paths = 500
        y = []
        for i in range(len(x)):
            totalDecrease = 0
            self.updateSec1Mean(x[i])
            np.random.seed(seed)
            for j in range(paths):
                pricemovements = self.price_move(periods)
                decreaseReturn = self.decrease_return(pricemovements, tcost, periods)
                totalDecrease += decreaseReturn * 100
            meanDecrease = np.round(totalDecrease / paths, 1)
            y.append(meanDecrease)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        image, = ax.plot(x, y)
        plt.ylabel('transaction cost (%)')
        plt.xlabel('security 1 return')
        plt.title('sec1mean - tcost graph')
        plt.draw()
        fig.savefig('sec1mean-tcost graph')
        print(
            'sec1mean-tcost:\nseed=%d\nsec1vol=%.2f\nsec2vol=%.2f\ncorr=%.2f\nsec1mean=%.2f-%.2f\nsec2mean=%.2f\nthreshold=%.2f'
            % (seed, self.sec1_volume, self.sec2vol, self.corr, start, end, self.means[1], self.rebalance_threshold))
        print('coef:', np.polyfit(x, y, 1))

    def solve_threshold(self, paths, tcost, periods, seed):
        start = 1
        end = 10
        x = np.linspace(1, 10, 11)
        y = []
        for i in range(len(x)):
            totalDecrease = 0
            self.update_threshold(x[i] / 100)
            np.random.seed(seed)
            for j in range(paths):
                pricemovements = self.price_move(periods)
                decreaseReturn = self.decrease_return(pricemovements, tcost, periods)
                totalDecrease += decreaseReturn * 100
            meanDecrease = np.round(totalDecrease / paths, 1)
            y.append(meanDecrease)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        image, = ax.plot(x, y)
        plt.ylabel('transaction cost (%)')
        plt.xlabel('rebalance threshold (%)')
        plt.title('threshold - tcost graph')
        plt.draw()
        fig.savefig("threshold-tcost graph")
        print(
            "threshold-tcost:\nseed=%d\nsec1vol=%.2f\nsec2vol=%.2f\ncorr=%.2f\nsec1mean=%.2f\nsec2mean=%.2f\nthreshold=%.2f-%.2f"
            % (seed, self.sec1_volume, self.sec2vol, self.corr, self.means[0], self.means[1], start, end))
        print('coef:', np.polyfit(x, y, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sec1vol", help="annualized volatility of security 1", type=float, default=.4)
    parser.add_argument("--sec2vol", help="annualized volatility of security 2", type=float, default=.3)
    parser.add_argument("--corr", help="correlation between security 1 and 2", type=float, default=.8)
    parser.add_argument("--sec1mean", help="annualized return of security 1", type=float, default=.05)
    parser.add_argument("--sec2mean", help="annualized return of security 2", type=float, default=.1)
    parser.add_argument("--paths", help="number of monte carlo iterations", type=int, default=500)
    parser.add_argument("--periods", help="number of days", type=int, default=252)
    parser.add_argument("--tcost", help="transaction cost per trade", type=float, default=.1)
    parser.add_argument("--rebalance_threshold", help="the minimal divergence that causes rebalance", type=float,
                        default=.01)
    parser.add_argument("--seed", help="set seed for the simulation", type=int, default=5)
    parser.add_argument("--simulate",
                        help="plot price movements of two stocks and print information about their transaction costs",
                        type=bool, default=False)
    parser.add_argument("--convergence_test", help="test convergence of transaction cost", type=bool, default=False)
    parser.add_argument("--step", help="set the step for convergence test", type=int, default=10)
    parser.add_argument("--solveCorr", help="solve transaction cost with respect to correlation coefficient", type=bool,
                        default=False)
    parser.add_argument("--solveVol", help="solve transaction cost with respect to the volatity of a security",
                        type=bool, default=False)
    parser.add_argument("--solveReturn", help="solve transaction cost with respect to the return of a security",
                        type=bool, default=False)
    parser.add_argument("--solveThreshold", help="solve transaction cost with respect to the rebalance threshold",
                        type=bool, default=False)
    parser.add_argument("--doAll", help="do everything",
                        type=bool, default=False)
    args = parser.parse_args()
    portfolio = Portfolio(args.sec1mean, args.sec2mean,
                          args.sec1vol, args.sec2vol, args.corr, args.rebalance_threshold)
    if args.simulate:
        portfolio.simulate(args.paths, args.tcost, args.periods, args.seed)
    elif args.convergence_test:
        portfolio.tests(args.paths, args.tcost, args.periods, args.step, args.seed)
    elif args.solveCorr:
        portfolio.solveCorr(args.paths, args.tcost, args.periods, args.seed)
    elif args.solveVol:
        portfolio.solveSec1Vol(args.paths, args.tcost, args.periods, args.seed)
    elif args.solveThreshold:
        portfolio.solve_threshold(args.paths, args.tcost, args.periods, args.seed)
    elif args.solveReturn:
        portfolio.solveSec1Mean(args.paths, args.tcost, args.periods, args.seed)
    elif args.doAll:
        portfolio.simulate(args.paths, args.tcost, args.periods, args.seed)
        portfolio.tests(args.paths, args.tcost, args.periods, args.step, args.seed)
        portfolio.solveCorr(args.paths, args.tcost, args.periods, args.seed)
        portfolio.solveSec1Vol(args.paths, args.tcost, args.periods, args.seed)
        portfolio.solve_threshold(args.paths, args.tcost, args.periods, args.seed)
        portfolio.solveSec1Mean(args.paths, args.tcost, args.periods, args.seed)


if __name__ == '__main__':
    main()
