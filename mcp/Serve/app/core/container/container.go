package container

import (
	"log"
	"strings"
	"sync"
)

var containerMap sync.Map

type Container struct{}

// CreateContainersFactory 创建一个容器工厂（interface）
func CreateContainersFactory() *Container {
	return &Container{}
}

// Set  1.设置
func (c *Container) Set(key string, fn interface{}) (res bool) {
	if _, exists := c.KeyIsExists(key); exists == false {
		containerMap.Store(key, fn)
		res = true
	} else {
		log.Fatal("键名重复,请解决键名重复问题,相关键：" + key)
	}
	return
}

// Delete  2.删除
func (c *Container) Delete(key string) {
	containerMap.Delete(key)
}

// Get 3.传递键，从容器获取值
func (c *Container) Get(key string) interface{} {
	if value, exists := c.KeyIsExists(key); exists {
		return value
	}
	return nil
}

func (c *Container) KeyIsExists(key string) (interface{}, bool) {

	return containerMap.Load(key)
}

// FuzzyDelete 根据前缀模糊匹配删除key
func (c *Container) FuzzyDelete(keyPre string) {
	containerMap.Range(func(key, value interface{}) bool {
		if keyName, ok := key.(string); ok {
			if strings.HasPrefix(keyName, keyPre) {
				containerMap.Delete(keyName)
			}
		}
		return true
	})
}
