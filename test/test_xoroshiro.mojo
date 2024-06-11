from testing import *
from numojo.xoroshiro import *


# Comparison values generated by xoroshiro128p.c
def test_xoroshiro128plus():
    var rng = Xoroshiro128Plus(123456789)
    assert_equal(rng.next(), 11600598462132880306)
    rng.jump()
    assert_equal(rng.next(), 1701379146372425724)
    rng.long_jump()
    assert_equal(rng.next(), 13777089769771307997)


def test_xoroshiro128plusplus():
    var rng = Xoroshiro128PlusPlus(123456789)
    assert_equal(rng.next(), 7623752501701327506)
    rng.jump()
    assert_equal(rng.next(), 12010143370689778774)
    rng.long_jump()
    assert_equal(rng.next(), 10920250730979368610)


def test_xoroshiro128starstar():
    var rng = Xoroshiro128StarStar(123456789)
    assert_equal(rng.next(), 14327819035421001106)
    rng.jump()
    assert_equal(rng.next(), 18308038801205870198)
    rng.long_jump()
    assert_equal(rng.next(), 346507841849519571)
