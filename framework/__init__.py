from framework.framework import Framework
from framework.framwork_manager import FrameworkManager
from framework.LSSE import LSSE
from framework.LSyE import LSyE
from framework.loss import BiCELo

frameworks = {
    LSSE.framework_name(): LSSE,
    LSyE.framework_name(): LSyE,
}