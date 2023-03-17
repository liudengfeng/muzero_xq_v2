import pytest
import xqcpp


def test_wrong_move():
    b = xqcpp.XqBoard()
    # b.show_board()
    # 游戏结束后再执行触发错误
    # 2 代表进行中 1 红胜 0 和 -1 红负
    b.set_result(0)
    # print(b.reward())
    action = xqcpp.movestr2action("4948")
    with pytest.raises(IndexError):
        b.move(action)
