#импорт необходимых библиотек
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import tree
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')


class Node():
    # инициализация класса
    def __init__(self, name, deep):
        self.deep_ = deep
        self.leaf_ = False
        self.name_ = name

    # записать значение предсказания в узел
    def set_predict(self, C):
        self.leaf_ = True
        self.C = C

    # получить предсказание
    def get_predict(self):
        return self.C

    # записать предикат
    def set_predicat(self, j, t):
        self.j = j
        self.t = t

    # получить предикат
    def get_predicat(self):
        return self.j, self.t


# модель построения дерева
class DecisionTreeMy():
    # инициализация класса
    def __init__(self):
        self.graph = {}

    # обучение модели
    def fit(self, X, y, max_deep=3, min_object=2, m=None):
        self.X = X
        self.y = y
        self.min_object = min_object
        self.max_deep = max_deep
        if m is None:
            self.m = X.shape[1]
        else:
            self.m = m
        self.fit_node(X, y, Node("", 0))

    # рекурсивная функция обучения
    def fit_node(self, X, y, node):
        self.graph[node.name_] = node

        # проверка условия останова
        if node.deep_ >= self.max_deep or X.shape[0] <= self.min_object:  # нет условия - ошибка константного прогноза
            # запись прогноза (среднего значения оставшихся ответов) в узел
            node.set_predict(self.count_C(y))
        else:
            j, t = self.find_best_predicate(X, y)  # поиск предиката
            Xl, yl, Xr, yr = self.split(X, y, j, t)  # делим датасет по найденному лучшему предикату

            # создаем узлы для левого и правого поддерева и вызываем функцию рекурсивно
            node_left = Node(node.name_ + "l", node.deep_ + 1)
            node_left.set_predicat(j, t)
            node_right = Node(node.name_ + "r", node.deep_ + 1)
            node_right.set_predicat(j, t)

            self.fit_node(Xl, yl, node_left)  # левое поддерево
            self.fit_node(Xr, yr, node_right)  # правое поддерево

    # поиск лучшего предиката перебором по j и t
    def find_best_predicate(self, X, y):
        best_j = 0
        best_t = 0
        best_q = -10000000

        for j in np.random.choice(range(X.shape[1]), size=self.m, replace=False):
            for i in range(X.shape[0]):
                t = (
                X.T[j][i])  # + X.T[j][i + 1])/2 #перебор существующих вариантов показывает лучший результат чем среднее

                Xl, yl, Xr, yr = self.split(X, y, j, t)
                if Xl.shape[0] != 0 and Xr.shape[0] != 0:
                    q = (self.count_inpurity(y) - Xl.shape[0] * self.count_inpurity(yl) / X.shape[0]
                         - Xr.shape[0] * self.count_inpurity(yr) / X.shape[0])
                    if q > best_q:
                        best_q = q
                        best_j = j
                        best_t = t
        return best_j, best_t

    # деление датасета по предкату
    def split(self, X, y, j, t):
        if t is None:
            print('t')
        return X[X.T[j] <= t], y[X.T[j] <= t], X[X.T[j] > t], y[X.T[j] > t]  # left and right

    # выдача прогноза массиву входных векторов X
    def predict(self, X):
        return np.apply_along_axis(self.predict_one, 1, X)

    # прогноз по каждому отдельному вектору x
    def predict_one(self, x):
        path = ''
        for i in range(self.max_deep + 1):
            node = self.graph[path + 'l']
            j, t = node.get_predicat()
            if x[j] <= t:
                path += 'l'
            else:
                path += 'r'

            node_next = self.graph[path]
            if node_next.leaf_:
                return node_next.get_predict()

    # визуализация дерева
    def visualization(self):
        for node in self.graph.values():
            if node.name_ == '':
                pass
            else:
                s = ''
                j, t = node.get_predicat()
                for i in range(node.deep_ - 1):
                    s += '|   '
                s += '|--- feature_' + str(j)
                if node.name_[-1] == 'l':
                    s += ' <= '
                elif node.name_[-1] == 'r':
                    s += ' > '
                else:
                    s = 'BAD'
                s += str(round(t, 2))
                print(s)
                l = ''
                if node.leaf_:
                    for i in range(node.deep_):
                        l += '|   '
                    l += '|--- value ' + '[' + str(round(node.get_predict(), 2)) + ']'
                    print(l)

    # расчет прогноза
    def count_C(self, y):
        raise NotImplementedError

    # расчет информативности
    def count_inpurity(self, y):
        raise NotImplementedError


class DecisionTreeRegressorMy(DecisionTreeMy):
    # расчет прогноза средним значением из узла дерева
    def count_C(self, y):
        return np.mean(y)

        # расчет информативности (как дисперсия для mse)

    def count_inpurity(self, y):
        return np.var(y)


class DecisionTreeClassifierMy(DecisionTreeMy):

    # модификация метода fit для приема параметра criterion как критерий Джини или энтропийный
    def fit(self, X, y, max_deep=3, min_object=2, criterion='gini'):
        # инициализация дополнительного параметра criterion
        self.criterion = criterion
        # вызов функции родителя
        super().fit(X, y, max_deep, min_object)

    # расчет прогноза (значение максимально представленного класса)
    def count_C(self, y):
        ar, ind = np.unique(y, return_counts=True)
        return ar[ind.argmax()]

    # расчет информативности (для многоклассовой классификации расчет будет чуть сложнее см. формулы вверху)
    def count_inpurity(self, y):
        if self.criterion == 'gini':
            p_one = y[y == 1].shape[0] / y.shape[0]
            p_zero = y[y == 0].shape[0] / y.shape[0]
            return 1 - (p_one ** 2 + p_zero ** 2)
        elif self.criterion == 'entropy':
            p_one = y[y == 1].shape[0] / y.shape[0]
            if p_one == 0:
                p_one = 10 ** -10
            p_zero = y[y == 0].shape[0] / y.shape[0]
            if p_zero == 0:
                p_zero = 10 ** -10
            return -(p_one * np.log2(p_one) + p_zero * np.log2(p_zero))
        else:
            raise NameError('переменная criterion не может быть ' + self.criterion)