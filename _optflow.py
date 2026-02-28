import cv2
import numpy as np
import scipy


class ImDistort:
    """
    Класс, позволяющий многократно воспроизводить геометрическое искажение.
    """
    # Строит недеформированную координатную сетку:
    @staticmethod
    def create_meshgrid(shape):
        y = np.arange(shape[0], dtype=float)
        x = np.arange(shape[1], dtype=float)
        xv, yv = np.meshgrid(x, y)
        return np.dstack([xv, yv])

    def __init__(self, meshgrid=None):
        # Если координатная сетка не задана:
        if meshgrid is None:
            self.meshgrid = None

        else:
            meshgrid = np.array(meshgrid)

            # Если вместо координатной сетки дан её размер, то создаём её
            # без деформации:
            if meshgrid.ndim == 1:
                self.meshgrid = self.create_meshgrid(meshgrid)

            # Если координатная сетка передана, то запоминаем её:
            elif meshgrid.ndim == 3 and meshgrid.shape[-1] == 2:
                self.meshgrid = self.meshgrid

            else:
                raise ValueError('Неожиданный размер координатной сетки: ' +
                                 f'{meshgrid.shape}!')

    # Создаёт деформацию, повторяющую оптический поток:
    @classmethod
    def from_opt_flow(cls, flow):
        meshgrid = cls.create_meshgrid(flow.shape)
        cls(meshgrid - flow)

    # Применяем искажения к заданному изображению:
    def __call__(self, img, mask_mode=False, *args, **kwargs):
        # Округляем координаты пикселей, если используется режим маски:
        if mask_mode:
            meshgrid = np.round(self.meshgrid)
        else:
            meshgrid = self.meshgrid

        # Для map_coordinates нужен обратный порядок координат:
        meshgrid = [meshgrid[..., 1],
                    meshgrid[..., 0]]

        # Если изображение не имеет третьего измерения:
        ndim = img.ndim
        if ndim == 2:
            return scipy.ndimage.map_coordinates(img, meshgrid,
                                                 *args, **kwargs)

        # Если изображение имеет три измерения:
        elif ndim == 3:
            result = []
            for ch in range(img.shape[-1]):
                channel = img[..., ch]
                channel = scipy.ndimage.map_coordinates(channel, meshgrid,
                                                        *args, **kwargs)
                result.append(channel)
            return np.dstack(result)

        else:
            raise ValueError('Неожиданное число измерений изображения: ' +
                             f'{ndim}!')

    # Оценка искажений с помощью оптического потока:
    def from_example(cls, source, target):
        pass

    # Объединение искажений (некоммутативно!):
    def __add__(self, other):
        return type(self)(other(self.meshgrid))
    # dist2(dist1(img)) = (dist1 + dist2)(img)


class OptFlow:
    """
    Обвязка вокруг cv2.calcOpticalFlowFarneback.
    Позволяет выполнять различные операции на базе оптического потока.
    """
    def __init__(self,
                 pyr_scale=0.5,
                 levels=10,
                 winsize=40,
                 poly_n=5,
                 poly_sigma=1.1,
                 iterations=10,
                 flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.iterations = iterations
        self.flags = flags

    # Создаёт экземпляр класса ImDistort по оптическому потоку,
    # оцененному по двум изображениям:
    def __call__(self, img1, img2, init_flow=None):
        # Переводим цветные изображения в оттенки серого, если надо:
        if img1.ndim > 2 and img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if img2.ndim > 2 and img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # Копируем начальное состояние оптического потока перед
        # использованием, чтобы не допустить изменений в оригинальной
        # переменной:
        if init_flow is not None:
            init_flow = init_flow.copy()

        # Вычисляем сам оптический поток:
        flow = cv2.calcOpticalFlowFarneback(img1,
                                            img2,
                                            init_flow,
                                            pyr_scale=self.pyr_scale,
                                            levels=self.levels,
                                            winsize=self.winsize,
                                            poly_n=self.poly_n,
                                            poly_sigma=self.poly_sigma,
                                            iterations=self.iterations,
                                            flags=self.flags)

        # Создаём класс ImDistort с соответствующим искажением:
        return ImDistort.from_opt_flow(flow)

    def seq_flows(self, imgs, cum_sum=True, **mpmap_kwargs):
        """
        Вычисляем потоки между соседними кадрами видеопоследовательности.
        """
        # Рассчёт потока между каждой парой соседних кадров:
        flows = mpmap(self.__call__, imgs[:-1], imgs[1:], **mpmap_kwargs)

        # Если поток отстраивается от первого изображения:
        if cum_sum:

            # Инициируем нулями поток для первого кадра с самим собой:
            flow = np.zeros_like(flows[0])
            cum_flows = [flow]

            # Накапливаем сдвиги для следующих кадров:
            for dflow in flows:
                flow = flow + dflow
                cum_flows.append(flow)

            # Заменяем исходные потоки, потокам с накоплением:
            flows = cum_flows

        return flows

    def seq_apply_flow2img(self, img, cum_flows, **mpmap_kwargs):
        """
        Восстанавливает последовательность кадров, используя
        лишь первый кадр и последовательность опт. потоков.
        """
        return mpmap(self.apply_flow2img, [img] * len(cum_flows), cum_flows, **mpmap_kwargs)
