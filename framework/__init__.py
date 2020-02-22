from framework.framework import Framework
from framework.framwork_manager import FrameworkManager
from framework.LSSE import LSSE
from framework.LSyE import LSyE
from framework.LSeE import LSeE
from framework.SeE import SeE
from framework.LE import LE
from framework.loss import BiCELo

frameworks = {
    LSSE.framework_name(): LSSE,
    LSyE.framework_name(): LSyE,
    LSeE.framework_name(): LSeE,
    SeE.framework_name(): SeE,
    LE.framework_name(): LE,
}